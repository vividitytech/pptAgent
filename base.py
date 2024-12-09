r"""

# Nomenclature

| Prefix | Definition | Examples |
| --- | --- | --- |
| `vn.get_` | Fetch some data | [`vn.get_related_ddl(...)`][vanna.base.base.VannaBase.get_related_ddl] |
| `vn.add_` | Adds something to the retrieval layer | [`vn.add_question_sql(...)`][vanna.base.base.VannaBase.add_question_sql] <br> [`vn.add_ddl(...)`][vanna.base.base.VannaBase.add_ddl] |
| `vn.generate_` | Generates something using AI based on the information in the model | [`vn.generate_sql(...)`][vanna.base.base.VannaBase.generate_sql] <br> [`vn.generate_explanation()`][vanna.base.base.VannaBase.generate_explanation] |
| `vn.run_` | Runs code (SQL) | [`vn.run_sql`][vanna.base.base.VannaBase.run_sql] |
| `vn.remove_` | Removes something from the retrieval layer | [`vn.remove_training_data`][vanna.base.base.VannaBase.remove_training_data] |
| `vn.connect_` | Connects to a database | [`vn.connect_to_snowflake(...)`][vanna.base.base.VannaBase.connect_to_snowflake] |
| `vn.update_` | Updates something | N/A -- unused |
| `vn.set_` | Sets something | N/A -- unused  |

# Open-Source and Extending

Vanna.AI is open-source and extensible. If you'd like to use Vanna without the servers, see an example [here](https://vanna.ai/docs/postgres-ollama-chromadb/).

The following is an example of where various functions are implemented in the codebase when using the default "local" version of Vanna. `vanna.base.VannaBase` is the base class which provides a `vanna.base.VannaBase.ask` and `vanna.base.VannaBase.train` function. Those rely on abstract methods which are implemented in the subclasses `vanna.openai_chat.OpenAI_Chat` and `vanna.chromadb_vector.ChromaDB_VectorStore`. `vanna.openai_chat.OpenAI_Chat` uses the OpenAI API to generate SQL and Plotly code. `vanna.chromadb_vector.ChromaDB_VectorStore` uses ChromaDB to store training data and generate embeddings.

If you want to use Vanna with other LLMs or databases, you can create your own subclass of `vanna.base.VannaBase` and implement the abstract methods.

```mermaid
flowchart
    subgraph VannaBase
        ask
        train
    end

    subgraph OpenAI_Chat
        get_sql_prompt
        submit_prompt
        generate_question
        generate_plotly_code
    end

    subgraph ChromaDB_VectorStore
        generate_embedding
        add_question_sql
        add_ddl
        add_documentation
        get_similar_question_sql
        get_related_ddl
        get_related_documentation
    end
```

"""

import json
import os
import re
import sqlite3
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse

from exceptions import DependencyError, ImproperlyConfigured, ValidationError
from dataformat import TrainingPlan, TrainingPlanItem, TableMetadata
from utils import validate_config_path


class DataBase(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)

    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        if self.language is None:
            return ""

        return f"Respond in the {self.language} language."

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"


        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response


    def extract_python(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted python script.
        """

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        scripts = re.findall(r"```python\n(.*)```", llm_response, re.DOTALL)
        if scripts:
            py_script = scripts[-1]
            self.log(title="Extracted python", message=f"{py_script}")
            return py_script

        scripts = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if scripts:
            py_script = scripts[-1]
            self.log(title="Extracted python", message=f"{py_script}")
            return py_script

        return llm_response
    
    
    def extract_table_metadata(ddl: str) -> TableMetadata:
      """
        Example:
        ```python
        vn.extract_table_metadata("CREATE TABLE hive.bi_ads.customers (id INT, name TEXT, sales DECIMAL)")
        ```

        Extracts the table metadata from a DDL statement. This is useful in case the DDL statement contains other information besides the table metadata.
        Override this function if your DDL statements need custom extraction logic.

        Args:
            ddl (str): The DDL statement.

        Returns:
            TableMetadata: The extracted table metadata.
        """
      pattern_with_catalog_schema = re.compile(
        r'CREATE TABLE\s+(\w+)\.(\w+)\.(\w+)\s*\(',
        re.IGNORECASE
      )
      pattern_with_schema = re.compile(
        r'CREATE TABLE\s+(\w+)\.(\w+)\s*\(',
        re.IGNORECASE
      )
      pattern_with_table = re.compile(
        r'CREATE TABLE\s+(\w+)\s*\(',
        re.IGNORECASE
      )

      match_with_catalog_schema = pattern_with_catalog_schema.search(ddl)
      match_with_schema = pattern_with_schema.search(ddl)
      match_with_table = pattern_with_table.search(ddl)

      if match_with_catalog_schema:
        catalog = match_with_catalog_schema.group(1)
        schema = match_with_catalog_schema.group(2)
        table_name = match_with_catalog_schema.group(3)
        return TableMetadata(catalog, schema, table_name)
      elif match_with_schema:
        schema = match_with_schema.group(1)
        table_name = match_with_schema.group(2)
        return TableMetadata(None, schema, table_name)
      elif match_with_table:
        table_name = match_with_table.group(1)
        return TableMetadata(None, None, table_name)
      else:
        return TableMetadata()

    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False



    def _extract_python_code(self, markdown_string: str) -> str:
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # ----------------- Connect to Any Database to run the Generated SQL ----------------- #

    def connect_to_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        role: Union[str, None] = None,
        warehouse: Union[str, None] = None,
        **kwargs
    ):
        try:
            snowflake = __import__("snowflake.connector")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[snowflake]"
            )

        if username == "my-username":
            username_env = os.getenv("SNOWFLAKE_USERNAME")

            if username_env is not None:
                username = username_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake username.")

        if password == "mypassword":
            password_env = os.getenv("SNOWFLAKE_PASSWORD")

            if password_env is not None:
                password = password_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake password.")

        if account == "my-account":
            account_env = os.getenv("SNOWFLAKE_ACCOUNT")

            if account_env is not None:
                account = account_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake account.")

        if database == "my-database":
            database_env = os.getenv("SNOWFLAKE_DATABASE")

            if database_env is not None:
                database = database_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake database.")

        conn = snowflake.connector.connect(
            user=username,
            password=password,
            account=account,
            database=database,
            client_session_keep_alive=True,
            **kwargs
        )

        def run_sql_snowflake(sql: str) -> pd.DataFrame:
            cs = conn.cursor()

            if role is not None:
                cs.execute(f"USE ROLE {role}")

            if warehouse is not None:
                cs.execute(f"USE WAREHOUSE {warehouse}")
            cs.execute(f"USE DATABASE {database}")

            cur = cs.execute(sql)

            results = cur.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

            return df

        self.dialect = "Snowflake SQL"
        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True

    def connect_to_sqlite(self, url: str, check_same_thread: bool = False,  **kwargs):
        """
        Connect to a SQLite database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to.
            check_same_thread (str): Allow the connection may be accessed in multiple threads.
        Returns:
            None
        """

        # URL of the database to download

        # Path to save the downloaded database
        path = os.path.basename(urlparse(url).path)

        # Download the database if it doesn't exist
        if not os.path.exists(path):
            response = requests.get(url)
            response.raise_for_status()  # Check that the request was successful
            with open(path, "wb") as f:
                f.write(response.content)
            url = path
        else: #existed case
            url = path
        # Connect to the database
        conn = sqlite3.connect(
            url,
            check_same_thread=check_same_thread,
            **kwargs
        )

        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        self.dialect = "SQLite"
        self.run_sql = run_sql_sqlite
        self.run_sql_is_set = True

    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")

        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def connect_to_db():
            return psycopg2.connect(host=host, dbname=dbname,
                        user=user, password=password, port=port, **kwargs)


        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                        conn.rollback()
                        raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres


    def connect_to_mysql(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        try:
            import pymysql.cursors
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install PyMySQL"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your MySQL host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your MySQL database")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your MySQL user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your MySQL password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your MySQL port")

        conn = None

        try:
            conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=dbname,
                port=port,
                cursorclass=pymysql.cursors.DictCursor,
                **kwargs
            )
        except pymysql.Error as e:
            raise ValidationError(e)

        def run_sql_mysql(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    conn.ping(reconnect=True)
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except pymysql.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_mysql

    def connect_to_clickhouse(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        try:
            import clickhouse_connect
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install clickhouse_connect"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your ClickHouse host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your ClickHouse database")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your ClickHouse user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your ClickHouse password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your ClickHouse port")

        conn = None

        try:
            conn = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=user,
                password=password,
                database=dbname,
                **kwargs
            )
            print(conn)
        except Exception as e:
            raise ValidationError(e)

        def run_sql_clickhouse(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    result = conn.query(sql)
                    results = result.result_rows

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(results, columns=result.column_names)
                    return df

                except Exception as e:
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_clickhouse

    def connect_to_oracle(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        **kwargs
    ):

        """
        Connect to an Oracle db using oracledb package. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_oracle(
        user="username",
        password="password",
        dsn="host:port/sid",
        )
        ```
        Args:
            USER (str): Oracle db user name.
            PASSWORD (str): Oracle db user password.
            DSN (str): Oracle db host ip - host:port/sid.
        """

        try:
            import oracledb
        except ImportError:

            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn:
            dsn = os.getenv("DSN")

        if not dsn:
            raise ImproperlyConfigured("Please set your Oracle dsn which should include host:port/sid")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your Oracle db user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your Oracle db password")

        conn = None

        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs
            )
        except oracledb.Error as e:
            raise ValidationError(e)

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(';'): #fix for a known problem with Oracle db where an extra ; will cause an error.
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle

    def connect_to_bigquery(
        self,
        cred_file_path: str = None,
        project_id: str = None,
        **kwargs
    ):
        """
        Connect to gcs using the bigquery connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_bigquery(
            project_id="myprojectid",
            cred_file_path="path/to/credentials.json",
        )
        ```
        Args:
            project_id (str): The gcs project id.
            cred_file_path (str): The gcs credential file path
        """

        try:
            from google.api_core.exceptions import GoogleAPIError
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[bigquery]"
            )

        if not project_id:
            project_id = os.getenv("PROJECT_ID")

        if not project_id:
            raise ImproperlyConfigured("Please set your Google Cloud Project ID.")

        import sys

        if "google.colab" in sys.modules:
            try:
                from google.colab import auth

                auth.authenticate_user()
            except Exception as e:
                raise ImproperlyConfigured(e)
        else:
            print("Not using Google Colab.")

        conn = None

        if not cred_file_path:
            try:
                conn = bigquery.Client(project=project_id)
            except:
                print("Could not found any google cloud implicit credentials")
        else:
            # Validate file path and pemissions
            validate_config_path(cred_file_path)

        if not conn:
            with open(cred_file_path, "r") as f:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(f.read()),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            try:
                conn = bigquery.Client(
                    project=project_id,
                    credentials=credentials,
                    **kwargs
                )
            except:
                raise ImproperlyConfigured(
                    "Could not connect to bigquery please correct credentials"
                )

        def run_sql_bigquery(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                job = conn.query(sql)
                df = job.result().to_dataframe()
                return df
            return None

        self.dialect = "BigQuery SQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_bigquery

    def connect_to_duckdb(self, url: str, init_sql: str = None, **kwargs):
        """
        Connect to a DuckDB database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to. Use :memory: to create an in-memory database. Use md: or motherduck: to use the MotherDuck database.
            init_sql (str, optional): SQL to run when connecting to the database. Defaults to None.

        Returns:
            None
        """
        try:
            import duckdb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[duckdb]"
            )
        # URL of the database to download
        if url == ":memory:" or url == "":
            path = ":memory:"
        else:
            # Path to save the downloaded database
            print(os.path.exists(url))
            if os.path.exists(url):
                path = url
            elif url.startswith("md") or url.startswith("motherduck"):
                path = url
            else:
                path = os.path.basename(urlparse(url).path)
                # Download the database if it doesn't exist
                if not os.path.exists(path):
                    response = requests.get(url)
                    response.raise_for_status()  # Check that the request was successful
                    with open(path, "wb") as f:
                        f.write(response.content)

        # Connect to the database
        conn = duckdb.connect(path, **kwargs)
        if init_sql:
            conn.query(init_sql)

        def run_sql_duckdb(sql: str):
            return conn.query(sql).to_df()

        self.dialect = "DuckDB SQL"
        self.run_sql = run_sql_duckdb
        self.run_sql_is_set = True

    def connect_to_mssql(self, odbc_conn_str: str, **kwargs):
        """
        Connect to a Microsoft SQL Server database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            odbc_conn_str (str): The ODBC connection string.

        Returns:
            None
        """
        try:
            import pyodbc
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install pyodbc"
            )

        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install sqlalchemy"
            )

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine

        engine = create_engine(connection_url, **kwargs)

        def run_sql_mssql(sql: str):
            # Execute the SQL statement and return the result as a pandas DataFrame
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                conn.close()
                return df

            raise Exception("Couldn't run sql")

        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True
    def connect_to_presto(
        self,
        host: str,
        catalog: str = 'hive',
        schema: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        combined_pem_path: str = None,
        protocol: str = 'https',
        requests_kwargs: dict = None,
        **kwargs
    ):
      """
        Connect to a Presto database using the specified parameters.

        Args:
            host (str): The host address of the Presto database.
            catalog (str): The catalog to use in the Presto environment.
            schema (str): The schema to use in the Presto environment.
            user (str): The username for authentication.
            password (str): The password for authentication.
            port (int): The port number for the Presto connection.
            combined_pem_path (str): The path to the combined pem file for SSL connection.
            protocol (str): The protocol to use for the connection (default is 'https').
            requests_kwargs (dict): Additional keyword arguments for requests.

        Raises:
            DependencyError: If required dependencies are not installed.
            ImproperlyConfigured: If essential configuration settings are missing.

        Returns:
            None
      """
      try:
        from pyhive import presto
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host:
        host = os.getenv("PRESTO_HOST")

      if not host:
        raise ImproperlyConfigured("Please set your presto host")

      if not catalog:
        catalog = os.getenv("PRESTO_CATALOG")

      if not catalog:
        raise ImproperlyConfigured("Please set your presto catalog")

      if not user:
        user = os.getenv("PRESTO_USER")

      if not user:
        raise ImproperlyConfigured("Please set your presto user")

      if not password:
        password = os.getenv("PRESTO_PASSWORD")

      if not port:
        port = os.getenv("PRESTO_PORT")

      if not port:
        raise ImproperlyConfigured("Please set your presto port")

      conn = None

      try:
        if requests_kwargs is None and combined_pem_path is not None:
          # use the combined pem file to verify the SSL connection
          requests_kwargs = {
            'verify': combined_pem_path,  # 使用转换后得到的 PEM 文件进行 SSL 验证
          }
        conn = presto.Connection(host=host,
                                 username=user,
                                 password=password,
                                 catalog=catalog,
                                 schema=schema,
                                 port=port,
                                 protocol=protocol,
                                 requests_kwargs=requests_kwargs,
                                 **kwargs)
      except presto.Error as e:
        raise ValidationError(e)

      def run_sql_presto(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            sql = sql.rstrip()
            # fix for a known problem with presto db where an extra ; will cause an error.
            if sql.endswith(';'):
                sql = sql[:-1]
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except presto.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.run_sql_is_set = True
      self.run_sql = run_sql_presto

    def connect_to_hive(
        self,
        host: str = None,
        dbname: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        auth: str = 'CUSTOM',
        **kwargs
    ):
      """
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            host (str): The host of the Hive database.
            dbname (str): The name of the database to connect to.
            user (str): The username to use for authentication.
            password (str): The password to use for authentication.
            port (int): The port to use for the connection.
            auth (str): The authentication method to use.

        Returns:
            None
      """

      try:
        from pyhive import hive
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host:
        host = os.getenv("HIVE_HOST")

      if not host:
        raise ImproperlyConfigured("Please set your hive host")

      if not dbname:
        dbname = os.getenv("HIVE_DATABASE")

      if not dbname:
        raise ImproperlyConfigured("Please set your hive database")

      if not user:
        user = os.getenv("HIVE_USER")

      if not user:
        raise ImproperlyConfigured("Please set your hive user")

      if not password:
        password = os.getenv("HIVE_PASSWORD")

      if not port:
        port = os.getenv("HIVE_PORT")

      if not port:
        raise ImproperlyConfigured("Please set your hive port")

      conn = None

      try:
        conn = hive.Connection(host=host,
                               username=user,
                               password=password,
                               database=dbname,
                               port=port,
                               auth=auth)
      except hive.Error as e:
        raise ValidationError(e)

      def run_sql_hive(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except hive.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.run_sql_is_set = True
      self.run_sql = run_sql_hive

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.run_sql("SELECT * FROM my_table")
        ```

        Run a SQL query on the connected database.

        Args:
            sql (str): The SQL query to run.

        Returns:
            pd.DataFrame: The results of the SQL query.
        """
        raise Exception(
            "You need to connect to a database first by running vn.connect_to_snowflake(), vn.connect_to_postgres(), similar function, or manually set vn.run_sql"
        )

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        visualize: bool = True,  # if False, will not generate plotly code
        allow_llm_to_see_data: bool = False,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        """
        **Example:**
        ```python
        vn.ask("What are the top 10 customers by sales?")
        ```

        Ask Vanna.AI a question and get the SQL query that answers it.

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train Vanna.AI on the question and SQL query.
            visualize (bool): Whether to generate plotly code and display the plotly figure.

        Returns:
            Tuple[str, pd.DataFrame, plotly.graph_objs.Figure]: The SQL query, the results of the SQL query, and the plotly figure.
        """

        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            if print_results:
                return None
            else:
                return sql, None, None

        try:
            df = self.run_sql(sql)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)

            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)
            # Only generate plotly code if visualize is True
            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        try:
                            display = __import__(
                                "IPython.display", fromlist=["display"]
                            ).display
                            Image = __import__(
                                "IPython.display", fromlist=["Image"]
                            ).Image
                            img_bytes = fig.to_image(format="png", scale=2)
                            display(Image(img_bytes))
                        except Exception as e:
                            fig.show()
                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None, None
        return sql, df, fig

  