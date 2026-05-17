#!/usr/bin/env python3
"""
Diagnostic: inspect ChatGPT DOM inside an existing conversation with messages.
"""

import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.browser.manager import BrowserManager
from src.config import Config


async def main():
    browser = BrowserManager()
    try:
        page = await browser.start()

        # Navigate to first existing conversation from sidebar
        await browser.navigate(Config.CHATGPT_URL)
        await asyncio.sleep(5)

        # Click first conversation in sidebar
        first_thread = await page.query_selector("nav a[href^='/c/']")
        if first_thread:
            href = await first_thread.get_attribute("href")
            print(f"Navigating to existing thread: {href}")
            await page.goto(f"https://chatgpt.com{href}", wait_until="domcontentloaded")
            await asyncio.sleep(5)
        else:
            print("No existing threads found, staying on home page")

        print(f"\nURL: {page.url}\n")

        # Deep DOM dump - get full structure of the conversation area
        print("=== FULL MAIN CONTENT STRUCTURE ===\n")
        structure = await page.evaluate("""
            () => {
                function describeElement(el, depth, maxDepth) {
                    if (depth > maxDepth) return [];
                    const results = [];
                    const indent = '  '.repeat(depth);
                    const tag = el.tagName?.toLowerCase() || '?';
                    const id = el.id ? `#${el.id}` : '';
                    const cls = el.className && typeof el.className === 'string' ? '.' + el.className.split(' ').filter(c => c).slice(0, 3).join('.') : '';
                    const role = el.getAttribute?.('role') ? ` role="${el.getAttribute('role')}"` : '';
                    const dataAttrs = Array.from(el.attributes || [])
                        .filter(a => a.name.startsWith('data-'))
                        .map(a => ` ${a.name}="${a.value}"`)
                        .join('');
                    const aria = Array.from(el.attributes || [])
                        .filter(a => a.name.startsWith('aria-'))
                        .map(a => ` ${a.name}="${a.value.substring(0, 50)}"`)
                        .join('');
                    
                    let text = '';
                    if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
                        text = ` "${el.textContent.trim().substring(0, 60)}"`;
                    }
                    
                    results.push(`${indent}<${tag}${id}${cls}${role}${dataAttrs}${aria}>${text}`);
                    
                    for (const child of el.children || []) {
                        results.push(...describeElement(child, depth + 1, maxDepth));
                    }
                    return results;
                }
                
                const main = document.querySelector('main');
                if (!main) return ['No <main> element found'];
                return describeElement(main, 0, 8);
            }
        """)
        for line in structure[:200]:
            print(line)
        if len(structure) > 200:
            print(f"\n... ({len(structure)} total lines, showing first 200)")

        # Specifically find conversation turns / messages
        print("\n\n=== MESSAGE/TURN DETECTION ===\n")
        turns = await page.evaluate("""
            () => {
                // Look for any elements that might be conversation turns
                const candidates = [];
                
                // Strategy 1: Look for role attributes
                document.querySelectorAll('[data-message-author-role]').forEach(el => {
                    candidates.push({strategy: 'data-message-author-role', tag: el.tagName, role: el.getAttribute('data-message-author-role'), text: (el.innerText||'').substring(0, 100)});
                });
                
                // Strategy 2: Look for article elements
                document.querySelectorAll('article').forEach(el => {
                    candidates.push({strategy: 'article', tag: 'article', text: (el.innerText||'').substring(0, 100)});
                });
                
                // Strategy 3: Look for turn-related classes
                const turnSelectors = [
                    '[class*="turn"]', '[class*="message"]', '[class*="conversation"]',
                    '[class*="chat"]', '[class*="response"]', '[class*="assistant"]',
                    '[class*="user"]', '[class*="human"]'
                ];
                for (const sel of turnSelectors) {
                    document.querySelectorAll(sel).forEach(el => {
                        candidates.push({
                            strategy: sel, 
                            tag: el.tagName, 
                            class: el.className?.substring?.(0, 120),
                            text: (el.innerText||'').substring(0, 80),
                            childCount: el.children.length,
                        });
                    });
                }
                
                // Strategy 4: Look for role="log", role="list" or role-related
                document.querySelectorAll('[role]').forEach(el => {
                    const role = el.getAttribute('role');
                    if (['log', 'list', 'listitem', 'article', 'region', 'group'].includes(role)) {
                        candidates.push({
                            strategy: `role=${role}`, 
                            tag: el.tagName, 
                            class: el.className?.substring?.(0, 80),
                            childCount: el.children.length,
                            text: (el.innerText||'').substring(0, 60),
                        });
                    }
                });
                
                // Strategy 5: Look for data-testid containing 'conversation' or 'message' or 'turn'
                document.querySelectorAll('[data-testid]').forEach(el => {
                    const testid = el.getAttribute('data-testid');
                    if (testid.match(/message|turn|conversation|thread|chat/i)) {
                        candidates.push({
                            strategy: 'data-testid', 
                            tag: el.tagName, 
                            testid,
                            text: (el.innerText||'').substring(0, 80),
                        });
                    }
                });
                
                return candidates;
            }
        """)
        for t in turns:
            print(f"  {json.dumps(t)}")
        if not turns:
            print("  No conversation turn elements found via any strategy!")

        # Find all buttons in the conversation area
        print("\n\n=== BUTTONS IN MAIN AREA ===\n")
        buttons = await page.evaluate("""
            () => {
                const main = document.querySelector('main');
                if (!main) return [];
                const btns = main.querySelectorAll('button');
                return Array.from(btns).map(b => ({
                    text: (b.textContent || '').trim().substring(0, 60),
                    ariaLabel: b.getAttribute('aria-label'),
                    dataTestid: b.getAttribute('data-testid'),
                    className: (b.className || '').substring(0, 80),
                    title: b.getAttribute('title'),
                    allAttrs: Array.from(b.attributes).map(a => `${a.name}="${a.value.substring(0, 40)}"`).join(', '),
                }));
            }
        """)
        for b in buttons:
            print(f"  [{b.get('dataTestid') or b.get('ariaLabel') or b.get('text', '?')[:30]}] attrs: {b['allAttrs']}")

        # Find the send button area
        print("\n\n=== COMPOSER/INPUT AREA ===\n")
        composer = await page.evaluate("""
            () => {
                const textarea = document.querySelector('#prompt-textarea');
                if (!textarea) return {error: 'no textarea found'};
                
                // Walk up to find the form/container
                let container = textarea;
                for (let i = 0; i < 10; i++) {
                    container = container.parentElement;
                    if (!container) break;
                }
                
                if (!container) return {error: 'no container found'};
                
                // Find all buttons near the textarea
                const buttons = container.querySelectorAll('button');
                return {
                    containerTag: container.tagName,
                    containerClass: (container.className || '').substring(0, 100),
                    buttons: Array.from(buttons).map(b => ({
                        text: (b.textContent || '').trim().substring(0, 50),
                        ariaLabel: b.getAttribute('aria-label'),
                        dataTestid: b.getAttribute('data-testid'),
                        type: b.getAttribute('type'),
                        allAttrs: Array.from(b.attributes).map(a => `${a.name}="${a.value.substring(0, 50)}"`).join(', '),
                    })),
                };
            }
        """)
        print(json.dumps(composer, indent=2))

        # Now send "hi" and inspect DOM after response
        print("\n\n=== SENDING TEST MESSAGE ===\n")
        
        # Type into the input
        textarea = await page.query_selector('#prompt-textarea')
        if textarea:
            await textarea.focus()
            await page.keyboard.type("hi", delay=50)
            await asyncio.sleep(1)
            
            # Look for send button again after typing (it may appear dynamically)
            print("Looking for send button after typing...")
            send_check = await page.evaluate("""
                () => {
                    const allBtns = document.querySelectorAll('button');
                    const candidates = [];
                    for (const btn of allBtns) {
                        const aria = btn.getAttribute('aria-label') || '';
                        const testid = btn.getAttribute('data-testid') || '';
                        const text = (btn.textContent || '').trim();
                        if (aria.toLowerCase().includes('send') || 
                            testid.toLowerCase().includes('send') ||
                            text.toLowerCase().includes('send')) {
                            candidates.push({
                                ariaLabel: aria,
                                dataTestid: testid,
                                text: text.substring(0, 50),
                                allAttrs: Array.from(btn.attributes).map(a => `${a.name}="${a.value.substring(0, 50)}"`).join(', '),
                            });
                        }
                    }
                    return candidates;
                }
            """)
            print(f"Send button candidates: {json.dumps(send_check, indent=2)}")
            
            # Try pressing Enter to send
            await page.keyboard.press("Enter")
            print("Pressed Enter to send message")
            
            # Wait for response
            print("Waiting 15 seconds for response...")
            await asyncio.sleep(15)
            
            # Now inspect the conversation DOM
            print("\n=== POST-RESPONSE DOM INSPECTION ===\n")
            
            post_turns = await page.evaluate("""
                () => {
                    const candidates = [];
                    
                    // Check all elements with data-message-author-role
                    document.querySelectorAll('[data-message-author-role]').forEach(el => {
                        candidates.push({
                            strategy: 'data-message-author-role',
                            tag: el.tagName,
                            role: el.getAttribute('data-message-author-role'),
                            text: (el.innerText||'').substring(0, 150),
                            id: el.id,
                            className: (el.className||'').substring(0, 80),
                        });
                    });
                    
                    // Check articles
                    document.querySelectorAll('article').forEach((el, i) => {
                        candidates.push({
                            strategy: 'article',
                            index: i,
                            text: (el.innerText||'').substring(0, 150),
                            childTags: Array.from(el.children).map(c => c.tagName).join(', '),
                            allAttrs: Array.from(el.attributes).map(a => `${a.name}="${a.value.substring(0, 40)}"`).join(', '),
                        });
                    });
                    
                    // Check for turn/message patterns
                    const selectors = [
                        '[class*="turn"]', '[class*="message"]', 
                        '[data-message-id]', '[data-scroll-anchor]',
                        '[data-testid*="conversation"]', '[data-testid*="turn"]',
                    ];
                    for (const sel of selectors) {
                        try {
                            document.querySelectorAll(sel).forEach(el => {
                                candidates.push({
                                    strategy: sel,
                                    tag: el.tagName,
                                    class: (el.className||'').substring(0, 80),
                                    dataAttrs: Array.from(el.attributes).filter(a => a.name.startsWith('data-')).map(a => `${a.name}=${a.value.substring(0, 40)}`).join(', '),
                                    text: (el.innerText||'').substring(0, 100),
                                });
                            });
                        } catch(e) {}
                    }
                    
                    return candidates;
                }
            """)
            for t in post_turns:
                print(f"  {json.dumps(t)}")
            if not post_turns:
                print("  No turn elements found!")
            
            # Dump ALL data-testid on page now
            print("\n=== ALL data-testid AFTER RESPONSE ===\n")
            post_testids = await page.evaluate("""
                () => {
                    const all = document.querySelectorAll('[data-testid]');
                    const seen = new Set();
                    const results = [];
                    for (const el of all) {
                        const testid = el.getAttribute('data-testid');
                        if (!seen.has(testid)) {
                            seen.add(testid);
                            results.push({
                                tag: el.tagName, 
                                testid, 
                                text: (el.textContent||'').trim().substring(0, 60),
                            });
                        }
                    }
                    return results;
                }
            """)
            for t in post_testids:
                print(f"  <{t['tag']}> data-testid=\"{t['testid']}\" text=\"{t['text'][:40]}\"")

            # Dump buttons with any copy/stop related attributes
            print("\n=== COPY/STOP/SEND BUTTONS AFTER RESPONSE ===\n")
            action_btns = await page.evaluate("""
                () => {
                    const allBtns = document.querySelectorAll('button');
                    const results = [];
                    for (const btn of allBtns) {
                        const allText = (btn.getAttribute('aria-label') || '') + ' ' + 
                                       (btn.getAttribute('data-testid') || '') + ' ' + 
                                       (btn.textContent || '') + ' ' +
                                       (btn.title || '');
                        const lower = allText.toLowerCase();
                        if (lower.match(/copy|stop|send|regenerat|continu|read aloud|thumb/)) {
                            results.push({
                                ariaLabel: btn.getAttribute('aria-label'),
                                dataTestid: btn.getAttribute('data-testid'),
                                text: (btn.textContent||'').trim().substring(0, 50),
                                title: btn.title,
                                allAttrs: Array.from(btn.attributes).map(a => `${a.name}="${a.value.substring(0, 60)}"`).join(', '),
                            });
                        }
                    }
                    return results;
                }
            """)
            for b in action_btns:
                print(f"  {json.dumps(b)}")

            # Deep structure dump of conversation area
            print("\n=== CONVERSATION AREA DEEP STRUCTURE ===\n")
            conv_structure = await page.evaluate("""
                () => {
                    function describeElement(el, depth, maxDepth) {
                        if (depth > maxDepth) return [];
                        const results = [];
                        const indent = '  '.repeat(depth);
                        const tag = el.tagName?.toLowerCase() || '?';
                        const id = el.id ? `#${el.id}` : '';
                        const cls = el.className && typeof el.className === 'string' ? '.' + el.className.split(' ').filter(c => c).slice(0, 3).join('.') : '';
                        const dataAttrs = Array.from(el.attributes || [])
                            .filter(a => a.name.startsWith('data-'))
                            .map(a => ` ${a.name}="${a.value.substring(0, 40)}"`)
                            .join('');
                        const role = el.getAttribute?.('role') ? ` role="${el.getAttribute('role')}"` : '';
                        const aria = el.getAttribute?.('aria-label') ? ` aria-label="${el.getAttribute('aria-label').substring(0, 40)}"` : '';
                        
                        let text = '';
                        if (el.childNodes.length <= 2 && el.textContent && el.textContent.trim().length < 60 && el.textContent.trim().length > 0 && el.children.length === 0) {
                            text = ` "${el.textContent.trim()}"`;
                        }
                        
                        results.push(`${indent}<${tag}${id}${cls}${role}${dataAttrs}${aria}>${text}`);
                        
                        for (const child of el.children || []) {
                            results.push(...describeElement(child, depth + 1, maxDepth));
                        }
                        return results;
                    }
                    
                    const main = document.querySelector('main');
                    if (!main) return ['No <main> found'];
                    return describeElement(main, 0, 10);
                }
            """)
            for line in conv_structure[:300]:
                print(line)
            if len(conv_structure) > 300:
                print(f"\n... ({len(conv_structure)} total lines, first 300 shown)")

        print("\n\nDone. Press ENTER to close...")
        input()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
