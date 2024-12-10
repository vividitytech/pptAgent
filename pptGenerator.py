from openAPI import OpenAIChat
from faiss_vector import FAISS
from base import DataBase
import pandas as pd
from pptx import Presentation
import csv
import subprocess
import re

class PPTGenerator(DataBase):

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config=config.get("dbconfig", None))
        self.vec = FAISS(config=config.get("faissconfig", None))
        self.llm = OpenAIChat(config=config.get("chatconfig", None))
        

    def get_sqlite_table_info(self, cursor, conn):
        '''
        prints out all of the columns of every table in db
        cursor : cursor object
        conn : database connection object
        '''
        tab_indent = '\t'
        alltables = ""
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        for table_name in tables:
            table_name = table_name[0] # tables is a list of single item tuples
            table = pd.read_sql_query("SELECT * from {} LIMIT 0".format(table_name), conn)
            alltables =alltables + table_name + ":\n"
            alltables =alltables + format ("its column names splited by tab: " +(tab_indent.join(table.columns)))
            #for col in table.columns:
            #    print('\t' + col)
            #print ("\ncolumn names for %s:" % table_name)+": "+(tab_indent.join(table.columns))
            alltables =alltables + "\n\n"
        return alltables
    
    def sqlite_execute(self, query_str, cursor, conn, out_name = 'output.csv'):

        # a='''.read "scheduling.sql"'''
        cursor.execute(query_str)
        conn.commit
        results = cursor.fetchall()
        # Write the results to a CSV file
        with open(out_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([i[0] for i in cursor.description])  # Write header row
            writer.writerows(results)
        return results
    
    def chat2sql_call_llm(self, user_message, system_message, cursor, db, args):
        
        self.llm.set_system_content(system_message)
        
        for i in range(args.max_tries):
            try:
                sql_script = self.llm.chat(user_message)
                # sql_script = self.extract_sql('To retrieve the top 10 customers by sales from the provided database schema, you can use the following SQL query. This query joins the `Customer` and `Invoice` tables, calculates the total sales for each customer, and then orders the results to get the top 10 customers.\n\nHere is the SQL query:\n\n```sql\nSELECT \n    c.CustomerId,\n    c.FirstName,\n    c.LastName,\n    SUM(i.Total) AS TotalSales\nFROM \n    Customer c\nJOIN \n    Invoice i ON c.CustomerId = i.CustomerId\nGROUP BY \n    c.CustomerId\nORDER BY \n    TotalSales DESC\nLIMIT 10;\n```\n\n### Explanation:\n- The query selects the `CustomerId`, `FirstName`, and `LastName` from the `Customer` table.\n- It calculates the total sales for each customer by summing up the `Total` column from the `Invoice` table.\n- It joins the `Invoice` table to the `Customer` table using the `CustomerId`.\n- The results are grouped by `CustomerId` to ensure that each customer’s sales are aggregated.\n- It orders the results in descending order of total sales.\n- Finally, it limits the results to the top 10 customers.')
                # results = execute("SELECT \n    c.CustomerId,\n    c.FirstName,\n    c.LastName,\n    SUM(i.Total) AS TotalSales\nFROM \n    Customer c\nJOIN \n    Invoice i ON c.CustomerId = i.CustomerId\nGROUP BY \n    c.CustomerId\nORDER BY \n    TotalSales DESC\nLIMIT 10")
                sql_script = self.extract_sql(sql_script)
                results = self.sqlite_execute(sql_script, cursor, db, args.infile_csv)
                self.vec.add_question_sql(question=user_message, sql=sql_script)
                break
            except Exception as e:
                user_message = "I got follow Exception: \n" + str(e) +"\n" + "Please generate runnable script"

    def print_template_format(self, infile_ppt= "template.pptx", layout_index = 2):
        # Load your presentation template
        prs = Presentation(infile_ppt)

        # Get the layout
        slide_layout = prs.slide_layouts[layout_index]

        # Print placeholder details for the chosen layout
        formatOut = f"Details for layout {layout_index}: {slide_layout.name}"
        for placeholder in slide_layout.placeholders:
            formatOut = formatOut + "\n"
            formatOut = formatOut + f"Placeholder index: {placeholder.placeholder_format.idx}, Type: {placeholder.placeholder_format.type}, Name: '{placeholder.name}'"
        return formatOut

    def get_csv_struct(self, infile_name = "output.csv"):
        #df = pd.read_csv(infile_name)
        #csv_reader = csv.reader(infile_name)
        #header = next(csv_reader)  # Read the first row as header
        header = ""
        with open(infile_name, 'r') as f:
            dict_reader = csv.DictReader(f)

            #get header fieldnames from DictReader and store in list
            headers = dict_reader.fieldnames
            
            for h in headers:
                header = header + "\t" + h
        return header #df.head()  # Print the first few rows of the DataFrame
    

    def csv2ppt_call_llm(self,prompt, args):

        self.llm.set_system_content("You are expert in python and you can geenrate python script based on user guidence. The generated script should contain no other information exception script, which can save and run without further human interaction.")

        for i in range(args.max_tries):
            try:

                python_llm_response = self.llm.chat(prompt)
                python_script = self.extract_python(python_llm_response)
                with open("myscript.py", "w") as f:
                    f.write(python_script)
                subprocess.run(["python", 'myscript.py'], check = True, capture_output=True)
                break
            except subprocess.CalledProcessError as e:
                print ('wrongcommand does not exist'.format({}), e.stderr)
                #File "myscript.py"
                error_description = re.findall(r"\bmyscript.py\b.*?File", str(e.stderr), re.DOTALL)
                if error_description:
                    prompt = "I got follow Exception: \n" + error_description[-1] +" "
                error_type = re.findall(r"\braise \b.*?Error", str(e.stderr), re.DOTALL)
                if error_type:
                    prompt = prompt + error_type[-1]
                prompt = prompt +"\n" + "Please try again to generate runnable script"
        
