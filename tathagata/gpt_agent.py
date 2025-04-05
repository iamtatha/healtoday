from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import time
import sys
import json
from stopping_condition import ConversationMonitor
from evaluators import PerplexityMetric, ReadabilityMetric, EntailmentMetric, EmpathicDialogueEvaluator, BERTSimilarity
import pandas as pd
import os
from dotenv import load_dotenv
import openai
load_dotenv(override=True)


class GPTAgent:
    def __init__(self, therapist_model, patient_model):
        self.output_file_name = f"results/{therapist_model}_{patient_model}_conversation"
        self.output_txt_file = open(f"{self.output_file_name}.txt", 'w')
        self.output_json_file = open(f"{self.output_file_name}.json", 'w')
        api_key = os.getenv("OPENAI_KEY")
        org = os.getenv("OPENAI_ORG")
        openai.api_key = api_key
        openai.organization = org
        self.therapist_model = therapist_model
        self.patient_model = patient_model
        self.avg_metrics = {'total': 0}
        self.avg_therapist_metrics = {'total': 0}
        self.avg_patient_metrics = {'total': 0}
        self.getPrompts()
        self.getModels()
        self.startConversation()
        self.storeMetrics(therapist_model, patient_model)
        
    def storeMetrics(self, therapist_model, patient_model):
        filename = f"results/summary_metrics.xlsx"
        metrics = {"Model": f"{therapist_model}_{patient_model}"}
        for key in self.avg_metrics:
            metrics[key] = self.avg_metrics[key]
        
        try:
            prev = pd.read_excel(filename)
            metrics = {len(prev): metrics}
            metrics = pd.DataFrame(metrics).T
            df = pd.concat([prev, metrics], ignore_index = True)
        except FileNotFoundError:
            metrics = {0: metrics}
            metrics = pd.DataFrame(metrics).T
            df = metrics
        df.to_excel(filename, index = False)
        
    def getPrompts(self):
        with open("prompts/therapist.txt", "r") as file:
            self.therapist_prompt = file.read()
        with open("prompts/patient.txt", "r") as file:
            self.patient_prompt = file.read()
        with open("prompts/therapist_end.txt", "r") as file:
            self.therapist_end_prompt = file.read()

    def getModels(self):
        self.agent_1 = ChatOpenAI(model_name=self.therapist_model, temperature=0.7, openai_api_key=openai.api_key)
        self.agent_2 = ChatOpenAI(model_name=self.patient_model, temperature=0.7, openai_api_key=openai.api_key)
        self.memory_1 = ConversationBufferMemory()
        self.memory_2 = ConversationBufferMemory()

        self.conversation_1 = ConversationChain(llm = self.agent_1, memory = self.memory_1)
        self.conversation_2 = ConversationChain(llm = self.agent_2, memory = self.memory_2)

    def getMetrics(self, person, context, response):
        perplexity = PerplexityMetric()
        readability = ReadabilityMetric()
        entailment = EntailmentMetric()
        empathy = EmpathicDialogueEvaluator()
        bertsimilarity = BERTSimilarity()

        ppl_scores = perplexity.calculate_perplexity(response)
        readability_scores = readability.calculate_readability(response)
        entailment_scores = entailment.check_entailment(context, response)
        empathy_scores = empathy.full_pipeline(context, response)
        bertsim = bertsimilarity.calculate_bertscore(context, response)
        
        metrics = {'Coherence': {'Perplexity': ppl_scores, "Flesch Reading Ease": readability_scores["Flesch Reading Ease"], "Gunning Fog Index": readability_scores["Gunning Fog Index"]}}
        metrics['consistency'] = {"entailment_score": entailment_scores["entailment_score"], "neutral_score": entailment_scores["neutral_score"], "contradiction_score": entailment_scores["contradiction_score"], "prediction": entailment_scores["prediction"]}
        metrics['Empathy'] = {"emotion_alignment": empathy_scores["emotion_alignment"], "context_emotion": empathy_scores["context_emotion"], "response_emotion": empathy_scores["response_emotion"], "response_emotion_score": empathy_scores["response_emotion_score"]}
        metrics['Thought Similarity'] = {"BERT similarity": bertsim}
        
        try:
            self.avg_metrics['Perplexity'] += ppl_scores
            self.avg_metrics['Flesch Reading Ease'] += readability_scores["Flesch Reading Ease"]
            self.avg_metrics['Gunning Fog Index'] += readability_scores["Gunning Fog Index"]
            self.avg_metrics['Entailment-Neutral-Consistency'] += max({k: v for k, v in entailment_scores.items() if isinstance(v, (int, float))}.values())
            self.avg_metrics['Empathy'] += empathy_scores["emotion_alignment"]
            self.avg_metrics['BERTScore'] += bertsim
            
            if person == 0:
                self.avg_therapist_metrics['Perplexity'] += ppl_scores
                self.avg_therapist_metrics['Flesch Reading Ease'] += readability_scores["Flesch Reading Ease"]
                self.avg_therapist_metrics['Gunning Fog Index'] += readability_scores["Gunning Fog Index"]
                self.avg_therapist_metrics['Entailment-Neutral-Consistency'] += max({k: v for k, v in entailment_scores.items() if isinstance(v, (int, float))}.values())
                self.avg_therapist_metrics['Empathy'] += empathy_scores["emotion_alignment"]
                self.avg_therapist_metrics['BERTScore'] += bertsim
            else:
                self.avg_patient_metrics['Perplexity'] += ppl_scores
                self.avg_patient_metrics['Flesch Reading Ease'] += readability_scores["Flesch Reading Ease"]
                self.avg_patient_metrics['Gunning Fog Index'] += readability_scores["Gunning Fog Index"]
                self.avg_patient_metrics['Entailment-Neutral-Consistency'] += max({k: v for k, v in entailment_scores.items() if isinstance(v, (int, float))}.values())
                self.avg_patient_metrics['Empathy'] += empathy_scores["emotion_alignment"]
                self.avg_patient_metrics['BERTScore'] += bertsim
            
        except KeyError:
            self.avg_metrics['Perplexity'] = ppl_scores
            self.avg_metrics['Flesch Reading Ease'] = readability_scores["Flesch Reading Ease"]
            self.avg_metrics['Gunning Fog Index'] = readability_scores["Gunning Fog Index"]
            self.avg_metrics['Entailment-Neutral-Consistency'] = max({k: v for k, v in entailment_scores.items() if isinstance(v, (int, float))}.values())
            self.avg_metrics['Empathy'] = empathy_scores["emotion_alignment"]
            self.avg_metrics['BERTScore'] = bertsim
            
            if person == 0:
                self.avg_therapist_metrics['Perplexity'] = ppl_scores
                self.avg_therapist_metrics['Flesch Reading Ease'] = readability_scores["Flesch Reading Ease"]
                self.avg_therapist_metrics['Gunning Fog Index'] = readability_scores["Gunning Fog Index"]
                self.avg_therapist_metrics['Entailment-Neutral-Consistency'] = max({k: v for k, v in entailment_scores.items() if isinstance(v, (int, float))}.values())
                self.avg_therapist_metrics['Empathy'] = empathy_scores["emotion_alignment"]
                self.avg_therapist_metrics['BERTScore'] = bertsim
            else:
                self.avg_patient_metrics['Perplexity'] = ppl_scores
                self.avg_patient_metrics['Flesch Reading Ease'] = readability_scores["Flesch Reading Ease"]
                self.avg_patient_metrics['Gunning Fog Index'] = readability_scores["Gunning Fog Index"]
                self.avg_patient_metrics['Entailment-Neutral-Consistency'] = max({k: v for k, v in entailment_scores.items() if isinstance(v, (int, float))}.values())
                self.avg_patient_metrics['Empathy'] = empathy_scores["emotion_alignment"]
                self.avg_patient_metrics['BERTScore'] = bertsim
            
        self.avg_metrics['total'] += 1
        if person == 0:
            self.avg_therapist_metrics['total'] += 1
        else:
            self.avg_patient_metrics['total'] += 1
        
        return metrics
    
    def getAvgMetrics(self):
        for key in list(self.avg_metrics.keys()):
            if key != 'total':
                self.avg_metrics[key] = self.avg_metrics[key] / self.avg_metrics['total']
                self.avg_therapist_metrics[key] = self.avg_therapist_metrics[key] / self.avg_therapist_metrics['total']
                self.avg_patient_metrics[key] = self.avg_patient_metrics[key] / self.avg_patient_metrics['total']
     
    def startConversation(self):
        message = self.therapist_prompt
        patient_initial = self.conversation_2.predict(input = self.patient_prompt)
        self.memory_2.save_context({"input": self.patient_prompt}, {"output": patient_initial})
        monitor = ConversationMonitor()
        iteration = 0
        save_interval = 1
        responses = []

        while(True):
            # print(monitor.time_elapsed, monitor.stop_alert, monitor.force_stop)
            if monitor.force_stop:
                break
            
            if monitor.stop_alert:
                response_1 = self.conversation_1.predict(input=f"{self.therapist_end_prompt}{message}")
            else:
                response_1 = self.conversation_1.predict(input=message)
                
            response_1 = response_1.replace('\n\n', '\n')
            self.memory_1.save_context({"input": message}, {"output": response_1})
            self.output_txt_file.write(f"Therapist: \n{response_1}\n\n")
            monitor.checkStoppingCondition(response_1)
            
            
            # response_2 = input('Agent 2:')
            response_2 = self.conversation_2.predict(input=response_1)
            response_2 = response_2.replace('\n\n', '\n')
            self.memory_2.save_context({"input": response_1}, {"output": response_2})
            
            self.output_txt_file.write(f"Patient: \n{response_2}\n\n\n")
            monitor.checkStoppingCondition(response_2)
            
            json_file = self.output_json_file
            responses.append({'therapist': response_1, 'metric': self.getMetrics(0, message, response_1)})
            responses.append({'patient': response_2, 'metric': self.getMetrics(1, response_1, response_2)})
            
            if iteration % save_interval == 0:
                json.dump(responses, self.output_json_file, indent=4)
                responses = []
            
            message = response_2  
            print('Iteration Done', iteration+1)
            iteration += 1

            # print(f"Agent 1: {response_1}")
            # print(f"Agent 2: {response_2}")
        
        self.getAvgMetrics()
        json.dump(self.avg_metrics, self.output_json_file, indent=4)
        


if len(sys.argv) < 3:
    print("\n\nUsage: python3 gpt_agent.py <therapist_model_name> <patient_model_name>")
    print(f"Default is used | therapist_model: {therapist_model} | patient_model: {patient_model}\n\n")
    sys.exit(1)
else:
    therapist_model = sys.argv[1] 
    patient_model = sys.argv[2]
    
GPTAgent(therapist_model, patient_model)