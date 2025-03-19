import puppeteer, { Browser, Page } from "puppeteer-core";
import UserAgent from 'user-agents';

class BrowserModule {
    private browser: Browser | null;
    private page: Page | null;
    private path: string;

    constructor(chromePath: string) {
        this.browser = null;
        this.page = null;
        this.path = chromePath;
    }

    /**
     * Initializes the browser with default configuration
     */
    async init(): Promise<void> {
        try {
            console.log('Starting browser...');
            const browserPromise = puppeteer.launch({
                args: [
                    "--no-sandbox",
                    "--disable-setuid-sandbox"
                ],
                defaultViewport: {
                    width: 1280,
                    height: 800,
                },
                executablePath: this.path,
                headless: false,
            });

            const userAgent = new UserAgent();
            this.browser = await browserPromise;
            this.page = await this.browser.newPage();
            await this.page.evaluateOnNewDocument(() => {
                // @ts-ignore
                delete navigator.__proto__.webdriver;
    
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            });
            await this.page.setUserAgent(userAgent.toString());
            console.log('Browser started successfully!');
        } catch (error) {
            console.error('Error initializing browser:', error);
            throw error;
        }
    }

    /**
     * Navigates to a specified URL
     * @param url - The URL to navigate to
     */
    async navigateToPage(url: string): Promise<void> {
        if (!this.page) {
            throw new Error('Browser not initialized. Call `init()` first.');
        }

        try {
            const referrer = "https://www.google.com/";
            await this.page.setExtraHTTPHeaders({ Referer: referrer });
            await this.page.goto(url, {
                waitUntil: "networkidle0",
            });
        } catch (error) {
            console.error('Error navigating to page:', error);
            throw error;
        }
    }

    /**
     * Closes the browser instance
     */
    async close(): Promise<void> {
        if (this.browser) {
            await this.browser.close();
            this.browser = null;
            this.page = null;
        }
    }

    /**
     * Gets the current page instance
     */
    getPage(): Page | null {
        return this.page;
    }

    async getAllInteractableElements(): Promise<any[]> {
        if (!this.page) {
            throw new Error('Browser not initialized. Call `init()` first.');
        }
    
        try {
            const elements = await this.page.evaluate(() => {
                const interactables: any = [];
                
                // Get all elements
                const allElements = document.querySelectorAll('*');
                
                allElements.forEach((element, index) => {
                    // Check if element is visible
                    const style = window.getComputedStyle(element);
                    const isVisible = style.display !== 'none' && 
                                    style.visibility !== 'hidden' && 
                                    style.opacity !== '0';
    
                    if (!isVisible) return;
    
                    const rect = element.getBoundingClientRect();
                    // Only include elements that have dimensions
                    if (rect.width === 0 || rect.height === 0) return;
    
                    const hasClickListener = (element as HTMLElement).onclick !== null || 
                                          element.getAttribute('onclick') !== null;
                    const isLink = element.tagName.toLowerCase() === 'a' && (element as HTMLAnchorElement).href;
                    const isFormElement = ['input', 'textarea', 'select', 'button'].includes(
                        element.tagName.toLowerCase()
                    );
                    const hasInteractiveRole = ['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio']
                        .includes(element.getAttribute('role') ?? '');
    
                    if (hasClickListener || isLink || isFormElement || hasInteractiveRole) {
                        // Get the absolute position relative to the page (including scroll)
                        const absoluteRect = {
                            x: rect.x + window.scrollX,
                            y: rect.y + window.scrollY,
                            width: rect.width,
                            height: rect.height,
                            top: rect.top + window.scrollY,
                            right: rect.right + window.scrollX,
                            bottom: rect.bottom + window.scrollY,
                            left: rect.left + window.scrollX,
                            centerX: rect.x + (rect.width / 2) + window.scrollX,
                            centerY: rect.y + (rect.height / 2) + window.scrollY
                        };
    
                        interactables.push({
                            tagName: element.tagName,
                            id: element.id || `element-${index}`,
                            // type: element.type || 'unknown',
                            text: element.textContent?.trim() || '',
                            href: element instanceof HTMLAnchorElement ? element.href : null,
                            value: element instanceof HTMLInputElement ? element.value : null,
                            isClickable: hasClickListener || isLink,
                            isInput: isFormElement,
                            role: element.getAttribute('role') || null,
                            bounds: absoluteRect,
                            isVisible: true,
                            zIndex: parseInt(style.zIndex) || 0
                        });
                    }
                });
    
                return interactables;
            });
    
            return elements;
        } catch (error) {
            console.error('Error getting interactable elements:', error);
            throw error;
        }
    }
    
    // Optional: Add a method to highlight elements
    async highlightElement(elementId: string): Promise<void> {
        if (!this.page) {
            throw new Error('Browser not initialized. Call `init()` first.');
        }
    
        await this.page.evaluate((id: string) => {
            const element = document.getElementById(id) || document.querySelector(`[data-element-id="${id}"]`);
            if (element) {
                const oldOutline = element.style.outline;
                const oldPosition = element.style.position;
                
                element.style.outline = '2px solid red';
                element.style.position = 'relative';
                
                // setTimeout(() => {
                //     element.style.outline = oldOutline;
                //     element.style.position = oldPosition;
                // }, 1000);
            }
        }, elementId);
    }
}

export default BrowserModule;