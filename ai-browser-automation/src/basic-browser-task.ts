import AiBrowserAutomation from "./AiBrowserAutomation";
import dotenv from "dotenv";

dotenv.config();


async function Main(){
    const aiBrowserAutomation = new AiBrowserAutomation();

    await aiBrowserAutomation.initializeLLM('HuggingFaceTB/SmolLM2-360M-Instruct')
    
    var response = await aiBrowserAutomation.promptModel("if i gave you a json list of html Nodes would you be able to tell me what element to click on given a goal?")
    
    console.log("answer: " + JSON.stringify(response))
}

Main()