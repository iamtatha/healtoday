import time
import math
from dotenv import load_dotenv
load_dotenv(override=True)

class ConversationMonitor:
    def __init__(self, time=300):
        self.wpm = 110                          # words spoken per minute
        self.pause_between = 5                  # pause between therapist question and answer
        self.time_elapsed = 0                   # total time elapsed since the beginning of the conversation
        self.conversation_limit = time           # total tentative time of conversation (in seconds)
        
        self.wps = self.wpm / 60
        self.conversation_stopper_time = self.conversation_limit * 0.9
        self.stop_alert = False
        self.force_stop = False
    
    def getDialogueTime(self, response):
        word_count = len(response.split())
        return math.ceil(word_count / self.wps)
    
    def checkStoppingCondition(self, response):
        self.time_elapsed += self.getDialogueTime(response) + self.pause_between
        if self.time_elapsed > self.conversation_limit * 1.2:
            if self.stop_alert == False:
                self.stop_alert = True
            else:
                self.force_stop = True
        if self.time_elapsed >= self.conversation_stopper_time:
            self.stop_alert = True
            return True
        return self.checkAllSymptomCriteria(response)
        
    def checkAllSymptomCriteria(self, response):
        return False                            # Yet to implement

conv = ConversationMonitor()
print(conv.checkStoppingCondition('Hey, what is going on?'))
print(conv.checkStoppingCondition('Just chilling. What about you? Anything new or interesting in life?'))
        