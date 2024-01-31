"""
This is a general framework for the blenderbot text generation model. 
Initalize the bot with a blenderbot model from https://huggingface.co/.
By default, the bot uses facebook/blenderbot-400M-distill. alternative model: facebook/blenderbot-3B
Use readInUtterances() to add to the conversation this isd done to keep context in the chatbot.
Use generateresponse() to generater a resonse to the last message
"""
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

class blenderBot:
    def __init__(self, modelName:str="facebook/blenderbot-400M-distill") -> None: #model names is a hugging face model
        self.name = modelName
        self.tokenizer = BlenderbotTokenizer.from_pretrained(modelName)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(modelName)
        self.chatHistory = [] #list of strings containging all the messages the bot has sent or recived

    def __repr__(self) -> str:
        return f"{self.name}"

    def readInUtterance(self, utterance:str):
        """
        Blender bot cant handle more then a few sentance (aproximatly 3 depending on thier length).
        Because of this we must check to see how many uterances there are in self.chathistory and remove remove the first group so there are only ever 3 utterances 
        """
        utterance = utterance.strip().split(".")
        self.chatHistory += utterance
        if len(self.chatHistory) > 3:
            self.chatHistory = self.chatHistory[-3:]

    def generateResponse(self) -> str:
        tokenInput = self.tokenizer("".join(self.chatHistory), return_tensors="pt")
        tokenResponse = self.model.generate(**tokenInput)
        response = self.tokenizer.decode(tokenResponse[0], skip_special_tokens=True)
        #print(f"blender bot:{response}")
        return response