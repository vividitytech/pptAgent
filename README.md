
# PPTGenerator

ppt agent which can generate powerpoint from user conversation. The key idea is how to create prompt to get reasonable answers from llm.

A Python implementation which can generate slides via conversation. It has two parts:

(1) chat to sql and then get structed data from database, such as sqlite, mysql, etc

    -- use faiss as vector db to find the simlar question and sql script pair to construct prompt for llm (refer to RAG)
    
    -- correct question and sql pair will be added to vector db (for future usage)

(2) strcuted data to powerpoint slides

    -- use llm to generate python script and call python-pptx to create slides
    
    -- retry will be used to guide llm to regenerate python script

### Usage

Here's how you'd instantiate PPTGenerator:

```python
from pptGenerator import PPTGenerator
from configs import CONFIGS as pptConfig
pptAgent = PPTGenerator(pptConfig)
```

And here's how you'd generate structued data from user conversation:

```python
pptAgent.chat2sql_call_llm(user_message, system_message, cursor, db, args)
```

And here's how you'd generate slides from structed data:

```python
pptAgent.csv2ppt_call_llm(prompt, args)
```


run the following script for test (you can change the user_message)

```python
python test_ppt_generator.py
```

### Example Data

Chinook.sqlite from Vanna

faiss vector data (for similar search)


### Samples

user message = "what are the top 10 customers by sales"
![result slide](output.pptx)


### Library Dependences
python-pptx
```
pip install python-pptx
```

faiss
```
pip install faiss-cpu or pip install faiss-gpu
```


### References

Code:

- [Vanna github](https://github.com/vanna-ai/vanna.git)
- [data to slides](https://analythical.com/blog/automating-powerpoint-slides-with-charts) 


### License

MIT
