import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from bert_score import BERTScorer
from typing import Dict, Tuple
import textstat
import math

class PerplexityMetric:
    def calculate_perplexity(self, text, model_name="gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return perplexity

class ReadabilityMetric:
    def calculate_readability(self, text):
        flesch_score = textstat.flesch_reading_ease(text)

        fog_index = textstat.gunning_fog(text)
        
        return {
            "Flesch Reading Ease": flesch_score,
            "Gunning Fog Index": fog_index
        }

class EntailmentMetric:
    def check_entailment(self, premise, hypothesis):
        model_name = "roberta-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = ["contradiction", "neutral", "entailment"]

        prediction = labels[torch.argmax(probs).item()]
        
        return {
            "entailment_score": probs[0][2].item(),
            "neutral_score": probs[0][1].item(),
            "contradiction_score": probs[0][0].item(),
            "prediction": prediction
        }

class EmpathicDialogueEvaluator:
    def __init__(self):
        self._load_models()
        
    def _load_models(self):
        """Load all required pretrained models"""
        # Empathic Dialogue Model (using a fine-tuned T5 for dialogue)
        self.edm = pipeline(
            "text2text-generation",
            model="microsoft/DialoGPT-medium",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Emotion classification model (BERT-based)
        self.emotion_model_name = "bert-base-uncased"
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            "bhadresh-savani/bert-base-uncased-emotion",
            num_labels=6
        )
        self.emotion_model.eval()
    
    def detect_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotions in text"""
        inputs = self.emotion_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        emotion_scores = {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}
        
        return emotion_scores
    
    def dot_product(self, dict1, dict2):
        return sum(dict1[k] * dict2[k] for k in dict1 if k in dict2)
    
    def evaluate_empathy(self, context: str, response: str) -> Dict[str, float]:
        """Evaluate how empathic a response is to the given context"""
        # Get context emotion
        context_emotion = self.detect_emotion(context)
        primary_context_emotion = max(context_emotion.items(), key=lambda x: x[1])[0]
        
        # Get response emotion
        response_emotion = self.detect_emotion(response)
        primary_response_emotion = max(response_emotion.items(), key=lambda x: x[1])[0]
        
        alignment = self.dot_product(context_emotion, response_emotion)
        emotion_alignment = response_emotion.get(primary_context_emotion, 0.0) * context_emotion.get(primary_context_emotion, 0.0)
        
        
        # print(context_emotion, response_emotion)
        # print(emotion_alignment)
        # print(self.dot_product(context_emotion, response_emotion))
        # print(context_emotion[primary_context_emotion] * response_emotion[primary_context_emotion])
        
        return {
            'context_emotion': primary_context_emotion,
            'response_emotion': primary_response_emotion,
            'response_emotion_score': response_emotion.get(primary_response_emotion, 0.0),
            'emotion_alignment': emotion_alignment
        }
    
    def full_pipeline(self, context: str, response: str) -> Tuple[str, Dict[str, float]]:
        """Complete pipeline from context to evaluated response"""
        evaluation = self.evaluate_empathy(context, response)
        return evaluation

class BERTSimilarity:
    def calculate_bertscore(self, context, response):
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([response], [context])
        return float(f"{F1.mean():.4f}")

    

# text = "The quick brown fox jumps over the lazy dog."
# perplexity = PerplexityMetric()
# ppl = perplexity.calculate_perplexity(text)
# print(f"Perplexity: {ppl:.2f}")

# readability = ReadabilityMetric()
# readability_scores = readability.calculate_readability(text)
# print(readability_scores)


# premise = "It's so sunny outside"
# hypothesis = "The sky is totally full of clouds"
# entailment = EntailmentMetric()
# result = entailment.check_entailment(premise, hypothesis)
# print(result)


# context = "I had an accident. I can't walk."
# response = "I am so sorry to hear that. is there anything I can do to help you?"
# evaluator = EmpathicDialogueEvaluator()
# evaluation = evaluator.full_pipeline(context, response)
# print(evaluation)