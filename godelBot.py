"""
This is a general framework for the godel text generation model. 
Initalize the bot with a godel model from https://huggingface.co/.
By default, the bot uses facebook/blenderbot-400M-distill. alternative model: facebook/blenderbot-3B
Use readInUtterances() to add to the conversation this isd done to keep context in the chatbot.
Use generateresponse() to generater a resonse to the last message
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class godelBot:
    def __init__(self, modelName:str="microsoft/GODEL-v1_1-base-seq2seq") -> None:
        self.name = modelName
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
        self.chatHistory = []
        self.chatHistoryLength = 0 #this is the number of tokens curently in the chat history
        self.knowledgeBase = "Dungeons & Dragons (D&D) is a tabletop role-playing game (RPG) where players create fictional characters and embark on adventures guided by a Dungeon Master (DM)."\
            "The game combines storytelling, strategy, and chance, using dice rolls to determine outcomes."\
                "Players explore imaginary worlds, interact with non-player characters, and overcome challenges, all while gaining experience and improving their characters' abilities."\
                    "D&D emphasizes creativity, cooperation, and improvisation, offering a flexible framework for collaborative storytelling and imaginative gameplay."
        self.instruction = "You are a player in a Dungeons and Dragons campaign."
    def readInUtterance(self, utterance:str):
        """
        The language model uses a colection of utterances in sequance to determine/generate a response. by using a chat history, context is kept in the conversation.
        The lm struggles to process any more then 500 tokens at a time so we must keep track of the number. This is a rough count as tokenization chnages the number words to tokens.
        if the number of tokens (chatHistotyLength) exceeds 500, we remove the first utterance of the chat history as it is unlikely it is still relevent. 
        """
        self.chatHistoryLength += len(utterance.strip().split())
        if self.chatHistoryLength >= 500: 
            self.chatHistory = self.chatHistory[1:]
        self.chatHistory.append(utterance)
    
    def generateResponse(self) -> str:
        knowledge = '[KNOWLEDGE] ' + self.knowledgeBase
        dialog = ' EOS '.join(self.chatHistory) #convert dialog into a string where each element is seperated by EOS
        query = f"{self.instruction} [CONTEXT] {dialog} {knowledge}"
        #print(f"Godel query:{query}")
        input_ids = self.tokenizer(f"{query}", return_tensors="pt").input_ids
        tokenResponse = self.model.generate(input_ids, max_length=128, min_length=10, top_p=0.9, do_sample=True) #(tokenized version of our information, max number of words in output text, min number of words in output text)
        return self.tokenizer.decode(tokenResponse[0], skip_special_tokens=True)
