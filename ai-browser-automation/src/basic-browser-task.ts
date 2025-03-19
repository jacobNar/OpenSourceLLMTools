import AiBrowserAutomation from "./AiBrowserAutomation";
import dotenv from "dotenv";
import BrowserModule from "./BrowserModule";

dotenv.config();


async function Main(){
    const aiBrowserAutomation = new AiBrowserAutomation();
    const browserModule = new BrowserModule("C:/Program Files/Google/Chrome/Application/chrome.exe");

    await aiBrowserAutomation.initializeLLM('HuggingFaceTB/SmolLM2-360M-Instruct')

    await browserModule.init();

    await browserModule.navigateToPage('https://www.google.com');

    const elements = await browserModule.getAllInteractableElements();
    
    // Highlight all elements simultaneously
    await Promise.all(elements.map(element => 
        browserModule.highlightElement(element.id)
    ));

    console.log(elements);
    
    var response = await aiBrowserAutomation.promptModel(`you are an AI agent that will be controlling a web browser. 
        I will give you a goal and an input of clickable elements from a web page, you will tell me the next acton to take. Actions you can take are 'move','click', 'type', 'scroll', 'wait', 'navigate' and 'close' followed by the element id. heres the webpage as a list of interactable elements: ${JSON.stringify(elements)}.`)
    
    console.log("answer: " + JSON.stringify(response))
}

Main()