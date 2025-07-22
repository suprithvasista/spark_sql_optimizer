import pandas as pd
import re
from pyspark.sql import SparkSession
from typing import List
from data_stats_skew import SparkRankProfiler  # Ensure this is available

class ETLProfilerFromExcel:
    def __init__(self, excel_path: str, db_name: str, spark: None):
        # Initialize Spark session inside the class if not provided
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("ETLProfilerFlow") \
                .enableHiveSupport() \
                .getOrCreate()
        else:
            self.spark = spark
        self.metadata_df = pd.read_excel(excel_path)
        # Convert all columns to string, strip whitespace, and keep original dtype if possible
        self.metadata_df = self.metadata_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        self.base_table = self.metadata_df['base_table'].iloc[0]
        self.db_name = db_name

        def strip_names(name: str) -> str:
            # Remove everything before the first underscore and after the last underscore
            name = name.lower()
            name = re.sub(r'^[^_]+_', '', name)   # Remove prefix before first underscore
            name = re.sub(r'_[^_]+$', '', name)   # Remove suffix after last underscore
            return name.strip("_")
            
        self.base_tag = strip_names(self.base_table)
        self.group_view = f"vw_{self.base_tag}_group_by"
        self.final_join_table = f"final_join_{self.base_tag}"
        self.final_dim_table = f"final_opti_{self.base_tag}"

        self.partition_value_filter = self._get_partition_filter()

    def _get_partition_filter(self) -> str:
        desc = self.spark.sql(f"DESCRIBE FORMATTED {self.db_name + '.' + self.base_table}").collect()
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

        partitions = self.spark.sql(f"SHOW PARTITIONS {self.db_name + '.' +self.base_table}").rdd.map(lambda r: r[0]).collect()
        latest = sorted(partitions)[-1]
        partition_filters = dict(part.split("=") for part in latest.split("/"))
        return ' AND '.join([f"{k} = '{v}'" for k, v in partition_filters.items()])

    def get_columns(self) -> List[str]:
        return self.metadata_df['join_column_source'].dropna().unique().tolist()

    def run(self):
        cols = self.get_columns()
        trimmed_cols=[v.strip() if isinstance(v, str) else v for v in cols]
        profiler = SparkRankProfiler(
            table_name=self.base_table,
            string_cols=trimmed_cols,
            db_name=self.db_name,
            yarn_limit_gb=None
        )
        summary_df, skewed_df = profiler.profile()
        skewed_cols = skewed_df['Column_Name'].tolist()

        print("✨ Profiling complete.")
        return [col.replace('_rank', '') for col in skewed_cols]

    def generate_and_execute_sql(self, skewed_cols: List[str]):
        safe = lambda c: f"nvl({c}, 'MISSING')"

        # Filter metadata to only rows with skewed columns in 'join_column_source'
        filtered_metadata = self.metadata_df[self.metadata_df['join_column_source'].isin(skewed_cols)]
        # NVL select and group by clauses for the skewed columns only
        nvl_select = ",\n  ".join([f"{safe(c)} AS {c}" for c in skewed_cols])
        nvl_group_by = ",\n  ".join([safe(c) for c in skewed_cols])

        base = self.base_table
        partition_filter = f"{self.partition_value_filter}"

        # Define concatenated string for final join key using skewed columns
        concatenated = " || '_' || ".join([f"nvl({col}, 'MISSING')" for col in skewed_cols])
        
        # Build joins and select columns based only on filtered metadata
        join_clauses = []
        select_columns = []
        concat_dim_concat = []

        for idx, row in filtered_metadata.iterrows():
            join_type = row.get('join_type')
            join_table = row.get('join_table')
            join_col_base = row.get('join_column_source')
            join_col_table = row.get('join_column_target')
            final_select_col = row.get('final_select_column')
            join_filter = row.get('filter_join')

            # Add join clause if all join keys present and join_table not base table
            if join_table and join_col_base and join_col_table and join_table != self.base_table:
                join_clause = f"{join_type} JOIN {join_table} ON {safe('cte.'+join_col_base)} = nvl({join_table}.{join_col_table}, 'MISSING') " + f'and {join_table}.' + join_filter if join_filter else ""
                if join_clause not in join_clauses:  # Avoid duplicates
                    join_clauses.append(join_clause)

            # Add final select column (likely the column from join table) if present
            if final_select_col:
                select_columns.append(join_table + '.' + final_select_col)
            concat_dim_concat.append(join_col_table)
        
        concatenated_dim = " || '_' || ".join([f"nvl({col}, 'MISSING')" for col in concat_dim_concat])
        # Join clauses as SQL
        join_sql_values = "\n  ".join(join_clauses)

        # Compose select clause: final select columns + all columns from base CTE (*)
        select_clause = ",\n  ".join(select_columns)
        
        # Filter metadata to only rows with NON-skewed columns
        non_skewed_metadata = self.metadata_df[~self.metadata_df['join_column_source'].isin(skewed_cols)]
        non_skewed_cols_filtered = non_skewed_metadata['join_column_source'].dropna().tolist()

        # Non-skewed NVL logic
        non_skewed_nvl_select = ",\n  ".join([f"{safe(c)} AS {c}" for c in non_skewed_cols_filtered])
        non_skewed_nvl_group_by = ",\n  ".join([safe(c) for c in non_skewed_cols_filtered])
        non_skewed_concatenated = " || '_' || ".join([f"nvl({col}, 'MISSING')" for col in non_skewed_cols_filtered])
        
        non_skewed_join_clauses = []
        non_skewed_select_columns = []
        
        updated_columns_for_final_sel = ['f.' + col.split('.', 1)[1] for col in select_columns]
        select_clause_final_query = ",\n  ".join(updated_columns_for_final_sel)
        
        for idx, row in non_skewed_metadata.iterrows():
            join_type = row.get('join_type')
            join_table = row.get('join_table')
            join_col_base = row.get('join_column_source')
            join_col_table = row.get('join_column_target')
            final_select_col = row.get('final_select_column')
            join_filter = row.get('filter_join')

            if join_table and join_col_base and join_col_table and join_table != self.base_table:
                join_clause = (
                    f"{join_type} JOIN {join_table} ON {safe('n.'+join_col_base)} = nvl({join_table}.{join_col_table}, 'MISSING') "
                    + (f"and {join_table}.{join_filter}" if join_filter else "")
                )
                if join_clause not in non_skewed_join_clauses:
                    non_skewed_join_clauses.append(join_clause)

            if final_select_col:
                non_skewed_select_columns.append(f"{join_table}.{final_select_col}")

        non_skewed_join_sql_values = "\n  ".join(non_skewed_join_clauses)
        non_skewed_select_clause = ",\n  ".join(non_skewed_select_columns)
        
        # Compose full SQL for GROUP BY table with joins
        group_sql = f"""
                    CREATE TABLE {self.group_view} AS
                    WITH cte AS (
                      SELECT
                        {nvl_select}
                      FROM {base}
                      WHERE {partition_filter}
                      GROUP BY
                        {nvl_group_by}
                    )
                    SELECT
                      {select_clause}
                    FROM cte
                    {join_sql_values}
                    """

        print(f"🔧 Running GROUP BY SQL:\n{group_sql}")
        #self.spark.sql(group_sql)

        # Final join intermediate table
        join_sql = f"""
        CREATE TABLE {self.final_join_table} AS
        SELECT
          {concatenated_dim} AS concated,
          *
        FROM {self.group_view}
        """
        print(f"🔧 Running JOIN SQL:\n{join_sql}")
        #self.spark.sql(join_sql)

        # Final dimension table
        dim_sql = f"""
        CREATE TABLE {self.final_dim_table} AS
        WITH new_ab AS (
          SELECT {concatenated} AS concated, *
          FROM {base}
          WHERE {partition_filter}
        )
        SELECT {select_clause_final_query} , {non_skewed_select_clause}
        FROM new_ab n
        LEFT JOIN {self.final_join_table} f
          ON n.concated = f.concated 
        {non_skewed_join_sql_values}
        """
        print(f"🔧 Running FINAL DIM SQL:\n{dim_sql}")
        #self.spark.sql(dim_sql)

        print("✅ SQL Pipeline executed successfully.")

if __name__ == "__main__":
    
    etl = ETLProfilerFromExcel(
        excel_path="sample_execution.xlsx",
        db_name="datbase",
        spark=None
    )

    skewed_columns = etl.run()
    etl.generate_and_execute_sql(skewed_columns)
