from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def car_facts(car_name: str):
   prompt_template = PromptTemplate(
      input_variables=["car_name"],
      template="Provide one positive and one negative feedback about the {car_name} received from car consumers?"
   )
   return prompt_template.format(car_name=car_name)

def llm_call(car_name: str):
   llm = ChatOpenAI(model_name="gpt-5.6-terra", temperature=0)
   prompt = car_facts(car_name)
   response = llm.invoke(prompt)
   return response.content

answer = llm_call("Toyota Camry")
print(answer)