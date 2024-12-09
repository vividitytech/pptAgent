import os
import csv
import sqlite3
import pandas as pd
from pptx import Presentation
import argparse
from pptGenerator import PPTGenerator
from configs import CONFIGS as pptConfig


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--sql_db_type", type=str, default="SQlite", help="what database to use, such mysql, sqlite"
)
argparser.add_argument(
    "--db_name",
    type=str,
    #required=True,
    help="database file name, which contains tables to query",
    default="Chinook.sqlite",
    choices=["Chinook.sqlite", "hotpotqa", "parallelqa"],
)
argparser.add_argument("--max_tries", type=int,  default=3 ,  help="maximum to try to query llm") #required=True,

argparser.add_argument("--do_benchmark", action="store_true", help="do benchmark")
argparser.add_argument(
    "--sleep_per_iter",
    type=int,
    default=None,
    help="Sleep seconds per iter to avoid rate limit",
)

# vllm-specific arguments
argparser.add_argument("--infile_csv", type=str, default="output.csv", help="file to save the query output from user")
argparser.add_argument("--infile_ppt", type=str, default="template.pptx", help="ppt template to use")
argparser.add_argument("--layout_index", type=int,  default=2,  help="layout index to use in template.ppt") #required=True,
argparser.add_argument("--outfile_ppt", type=str, default="output.pptx", help="ppt to generate")
args = argparser.parse_args()


# -------------------------------- user query ----------------------------------------
user_message = "what are the top 10 customers by sales"


pptAgent = PPTGenerator(pptConfig)

# ---------------------------------load db tables -------------------------------------
sql_db_type = args.sql_db_type
db = sqlite3.connect(args.db_name)
cursor = db.cursor()

# get tables' schema
infos = pptAgent.get_sqlite_table_info(cursor, db)
print(infos)
system_message = f"You are expert in {sql_db_type} and you can geenrate {sql_db_type} script based on user guidence (for example: user query sentence) and database schema. The generated script should output necessary columm names. You have the follow tables and you need to generate {sql_db_type} script which can run automatically\n"
system_message = system_message + infos

# system message for llm
similar_query = pptAgent.vec.get_similar_question_sql(user_message, n_results =1)
if similar_query:
    system_message = system_message + "\n" + f"""This is an example\nUser input: {similar_query[0]["question"]}""" + "\n" + f"""your response: {similar_query[0]["sql"]}""" +"\n"
# ---------------------------------- output data from user query --------------------------------------------
pptAgent.chat2sql_call_llm(user_message, system_message, cursor, db, args)

cursor.close()
db.close()


# ---------------------------------- construct prompt for ppt generation -----------------------------------
infile_csv = args.infile_csv
infile_ppt = args.infile_ppt

prompt = f"""
    I want you to create an auto-slides generator script, written in Python, that automatically generates PowerPoint slides with charts from an Excel file. To do this, you can use the existing python-pptx library to automate the slide creation and the pandas library for chart creation and formatting.
    
    You should insert chart to the slide by adding the corresponding values from the cvs file. Please remember to import pptx and chart types in the python script, sometime I see you forget to import correct Chart Type.

    When the auto-slides script is run, it should interrogate an csv data file named "{infile_csv}" and a PowerPoint template file named "{infile_ppt}". Both of these files will be located in the same directory as the main Python script. 

    The "{infile_csv}" file contains the following data structure:
    {pptAgent.get_csv_struct(infile_csv)}
    
    The "{infile_ppt}" file contains the following slide layout:
    {pptAgent.print_template_format(infile_ppt, args.layout_index)}
    
    The "{infile_ppt}" includes custom slide formatting and layouts to which the output file should adhere. Please use layout index "{args.layout_index}" to get the slide layout and insert values (int, float, etc) to the placeholder with CHART type.

    I want to set the title as "{user_message}", map the data from "{infile_csv}" to list and draw it as CHART. 
    """
# ------------------------------------ convert table data to ppt ---------------------------------------------
pptAgent.csv2ppt_call_llm(prompt, args)

# other test
for i in range(1, 1):
    pptAgent.vec.train(question=f"What are the total sales for customer {i}?", sql=f"SELECT SUM(sales) FROM example_sales WHERE customer_id = {i}")
output = pptAgent.vec.get_similar_question_sql("What are the total sales for customer for customer 1", n_results =10)
