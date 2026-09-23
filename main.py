from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def prompt_template(car_name: str):
   prompt_template = PromptTemplate(
      input_variables=["car_name"],
      template="Provide one positive and one negative feedback about the {car_name} received from car consumers?"
   )
   return prompt_template.format(car_name=car_name)
   

def llm_call(car_name: str):
   llm = ChatOpenAI(model_name="gpt-5.6-terra", temperature=0)
   prompt = prompt_template(car_name)
   response = llm.invoke(prompt)
   return response.content

#answer = llm_call("Toyota Tacoma TRD Pro")
#print(answer)

## to use chain and |
def prompt_template_chain():
    prompt_chain =  PromptTemplate(
      input_variables=["question"],
      template="You answer questions about Houston TX. {question}"
   )
    return prompt_chain

def llm_call_chain(question: str):
    #llm = ChatOpenAI(model_name="gpt-5.6-terra", temperature=0)
    #llm =  ChatOllama(model="gemma3:270m", temperature=0)
    llm =  ChatOllama(model="qwen3:4b-instruct", temperature=0)
    prompt_chain = prompt_template_chain()
    chain = prompt_chain | llm
    response = chain.invoke({"question": question})
    return response.content

answer_chain = llm_call_chain("what is the closest beach to City?")
print(answer_chain)


