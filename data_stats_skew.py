from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from functools import reduce
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np  # Make sure this is at the top of your script
from pyspark.sql.functions import hash, abs as abs_
from matplotlib.backends.backend_pdf import PdfPages

class SparkRankProfiler:
    def __init__(self, table_name: str, string_cols: list,  db_name: str , sample_ratio=0.01, yarn_limit_gb=None):
        self.table_name = table_name
        self.sample_ratio = sample_ratio
        self.string_cols = string_cols
        self.partition_filter = ""
        self.yarn_limit_gb = yarn_limit_gb  # Optional - can be dynamically inferred
        self.data_size_gb = 0
        self.batch_size = 0
        self.spark = None
        self.df_filtered = None
        self.df_ranked_all = None
        self.db_name = db_name

    def _init_pre_spark(self):
        return SparkSession.builder.appName("PreloadMemoryEstimate").enableHiveSupport().getOrCreate()

    def _get_partition_filter(self, spark):
        desc = spark.sql(f"DESCRIBE FORMATTED {self.db_name + '.' + self.table_name}").collect()
        partition_cols = []
        partition_section = False

        for row in desc:
            if row[0].strip() == "# Partition Information":
                partition_section = True
                continue
            if partition_section and row[0].strip() == "":
                break
            if partition_section:
                partition_cols.append(row[0].strip())

        partitions = spark.sql(f"SHOW PARTITIONS {self.db_name + '.' +self.table_name}").rdd.map(lambda r: r[0]).collect()
        latest = sorted(partitions)[-1]
        partition_filters = dict(part.split("=") for part in latest.split("/"))
        return ' AND '.join([f"{k} = '{v}'" for k, v in partition_filters.items()])

    def _estimate_data_size(self, spark):
        sample_df = spark.read.table(self.db_name + '.' +self.table_name).filter(self.partition_filter).select(self.string_cols).sample(self.sample_ratio)
        sample_bytes = sample_df.rdd.map(lambda row: len(str(row))).sum()
        total_est_bytes = sample_bytes / self.sample_ratio
        return total_est_bytes / (1024 ** 3)

    def _infer_yarn_limit_gb(self, spark):
        try:
            conf = spark.sparkContext.getConf()
            mem_str = conf.get("spark.executor.memoryOverhead", "512")  # MB
            exec_mem_str = conf.get("spark.executor.memory", "4096m")  # could be in MB or GB
            exec_mem_gb = int(exec_mem_str.lower().replace("g", "").replace("m", "")) / (1024 if "m" in exec_mem_str else 1)
            overhead_gb = int(mem_str) / 1024
            return exec_mem_gb + overhead_gb
        except Exception:
            return 52  # default fallback

    def _get_dynamic_config(self, data_size_gb, scale_factor=1.2):
        if self.yarn_limit_gb is None:
            self.yarn_limit_gb = self._infer_yarn_limit_gb(self.spark)

        base_mem = min(self.yarn_limit_gb * 0.75, max(4, math.ceil(data_size_gb * scale_factor)))
        overhead = self.yarn_limit_gb - base_mem
        return {
            "spark.executor.memory": f"{int(base_mem)}g",
            "spark.executor.memoryOverhead": f"{int(overhead * 1024)}",
            "spark.executor.cores": "4",
            "spark.sql.shuffle.partitions": str(min(400, max(100, int(data_size_gb * 10)))),
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.autoBroadcastJoinThreshold": "-1",
        }

    def _calculate_batch_size(self, data_size_gb):
        if data_size_gb <= 2:
            return 10
        elif data_size_gb <= 5:
            return 5
        elif data_size_gb <= 10:
            return 3
        else:
            return 2

    def _join_dfs_on_row_id(self, dfs):
        return reduce(lambda df1, df2: df1.join(df2, "row_id"), dfs)

    def profile(self):
        # Step 1: Preload Spark to get partition filter and estimate size
        pre_spark = self._init_pre_spark()
        self.partition_filter = self._get_partition_filter(pre_spark)
        print(f"Partition filter: {self.partition_filter}")

        self.data_size_gb = self._estimate_data_size(pre_spark)
        print(f"Estimated data size: {self.data_size_gb:.2f} GB")
        pre_spark.stop()

        # Step 2: Start Spark session with dynamic config
        self.spark = SparkSession.builder \
            .appName("EfficientStatsWithRankedStrings") \
            .enableHiveSupport() \
            .getOrCreate()

        config = self._get_dynamic_config(self.data_size_gb)
        driver_mem = max(4, math.ceil(self.data_size_gb * 0.1))

        # Apply configurations
        for k, v in config.items():
            self.spark.conf.set(k, v)
        self.spark.conf.set("spark.driver.memory", f"{driver_mem}g")

        self.batch_size = self._calculate_batch_size(self.data_size_gb)
        print(f"Using batch size: {self.batch_size}")

        # Step 3: Load and prepare data
        self.df_filtered = self.spark.read.table(self.db_name + '.' +self.table_name).filter(self.partition_filter).select(self.string_cols)
        self.df_filtered = self.df_filtered.withColumn("row_id", monotonically_increasing_id())
        self.df_filtered.persist()
        self.df_filtered.count()

        # Step 4: Hash-based pseudo-ranking (faster alternative to dense_rank)
        ranked_dfs = []
        for i in range(0, len(self.string_cols), self.batch_size):
            batch = self.string_cols[i:i + self.batch_size]
            df_batch = self.df_filtered.select("row_id", *batch)

            for col_name in batch:
                # Hash the string column to long (for numeric summary) using abs(hash)
                df_batch = df_batch.withColumn(f"{col_name}_rank", abs_(hash(col(col_name))).cast("long"))

            rank_cols = [f"{col_name}_rank" for col_name in batch]
            df_rank = df_batch.select("row_id", *rank_cols)
            df_rank.persist()
            df_rank.count()
            ranked_dfs.append(df_rank)

        # Step 5: Combine all rank-based DataFrames
        self.df_ranked_all = self._join_dfs_on_row_id(ranked_dfs)
        self.df_ranked_all.persist()
        self.df_ranked_all.count()

        # Step 6: Summary and transpose
        df_summary = self.df_ranked_all.summary()
        pdf = df_summary.toPandas()
        pdf_transposed = pdf.set_index('summary').transpose().reset_index()
        
        pdf_transposed.columns = ['Column_Name', 'count', 'mean', 'stddev', 'min', '25_PER', '50_PREC', '75_PREC', 'max']
        
        # Convert percentiles to numeric
        pdf_transposed[['25_PER', '50_PREC', '75_PREC']] = pdf_transposed[['25_PER', '50_PREC', '75_PREC']].apply(pd.to_numeric, errors='coerce')

        # Convert percentiles to numeric (safe)
        pdf_transposed[['25_PER', '50_PREC', '75_PREC', 'min', 'max']] = pdf_transposed[
            ['25_PER', '50_PREC', '75_PREC', 'min', 'max']
        ].apply(pd.to_numeric, errors='coerce')

        # Compute skewness
        pdf_transposed['iqr_skewness'] = (
            (pdf_transposed['75_PREC'] + pdf_transposed['25_PER'] - 2 * pdf_transposed['50_PREC']) /
            (pdf_transposed['75_PREC'] - pdf_transposed['25_PER'])
        )

        # Replace infinite values with NaN for safe comparison
        pdf_transposed['iqr_skewness'] = pdf_transposed['iqr_skewness'].replace([float('inf'), -float('inf')], np.nan)

        # Flag skewed columns
        skew_threshold = 0.5
        pdf_transposed['is_skewed'] = (
            pdf_transposed['iqr_skewness'].abs() > skew_threshold
        ) | (
            (pdf_transposed['75_PREC'] == pdf_transposed['25_PER']) &
            (pdf_transposed['max'] - pdf_transposed['50_PREC'] > 0)
        )

        # Get skewed columns separately if needed
        skewed_columns_df = pdf_transposed[pdf_transposed['is_skewed']]

        # Step 7: Cleanup
        for df in ranked_dfs:
            df.unpersist()
        self.df_filtered.unpersist()
        self.df_ranked_all.unpersist()

        return pdf_transposed, skewed_columns_df


class PercentilePlotter:
    def __init__(self, transposed_summary_df: pd.DataFrame):
        """
        Initialize with a transposed summary DataFrame.
        It must include columns: Column_Name, min, 25_PER, 50_PREC, 75_PREC, max.
        """
        self.df = transposed_summary_df.copy()
        self.required_columns = ['Column_Name', 'min', '25_PER', '50_PREC', '75_PREC', 'max']
        self._validate_input()

    def _validate_input(self):
        for col in self.required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        for col in self.required_columns[1:]:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _generate_histogram_from_percentiles(self, col_data, ax):
        """
        Simulate histogram using 5-point percentile data.
        """
        percentiles = [
            col_data['min'],
            col_data['25_PER'],
            col_data['50_PREC'],
            col_data['75_PREC'],
            col_data['max']
        ]
        bins = sorted(percentiles)
        synthetic_data = pd.Series(percentiles)

        sns.histplot(synthetic_data, bins=bins, kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {col_data["Column_Name"]}')
        ax.set_xlabel(col_data["Column_Name"])
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)

    def plot_all(self, output_pdf_path: str, plots_per_page: int = 10):
        """
        Plot histograms for all columns and save to a multi-page PDF.
        """
        num_columns = len(self.df)
        num_pages = (num_columns // plots_per_page) + (1 if num_columns % plots_per_page else 0)

        with PdfPages(output_pdf_path) as pdf:
            for page in range(num_pages):
                start_idx = page * plots_per_page
                end_idx = min(start_idx + plots_per_page, num_columns)
                subset = self.df.iloc[start_idx:end_idx]

                fig, axes = plt.subplots(nrows=len(subset), ncols=1, figsize=(10, 5 * len(subset)))
                if len(subset) == 1:
                    axes = [axes]  # make iterable if only one plot

                for i, (_, row) in enumerate(subset.iterrows()):
                    self._generate_histogram_from_percentiles(row, axes[i])

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"✅ PDF saved: {output_pdf_path}")        
        
#columns_to_profile = ['val1','val2']
#
#profiler = SparkRankProfiler(
#    table_name="db.table_name",
#    string_cols=columns_to_profile,
#    yarn_limit_gb=None  # Optional: use dynamic detection
#)
#
#summary_df, skewed_df = profiler.profile()
#summary_df.to_csv("ranked_summary_transposed.csv", index=False)
#skewed_df.to_csv("skewed_column_joins.csv", index=False)

#plotter = PercentilePlotter(transposed_summary_df=summary_df)
#plotter.plot_all(output_pdf_path="/path/percentile_distributions.pdf")
