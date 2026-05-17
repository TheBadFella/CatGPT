#!/usr/bin/env python3
"""
Diagnostic script: inspect the actual ChatGPT DOM to find working selectors.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.browser.manager import BrowserManager
from src.config import Config


async def main():
    browser = BrowserManager()
    try:
        page = await browser.start()
        await browser.navigate(Config.CHATGPT_URL)
        await asyncio.sleep(5)

        print("\n=== ChatGPT DOM Diagnostic ===\n")

        # 1. Check current URL
        print(f"URL: {page.url}\n")

        # 2. Check if logged in by looking for chat input
        checks = {
            # Chat input
            "#prompt-textarea": "Chat input (id)",
            "div[contenteditable='true'][id='prompt-textarea']": "Chat input (div contenteditable)",
            "div[contenteditable='true']": "Any contenteditable",
            # Send button
            "button[data-testid='send-button']": "Send button (testid)",
            "button[aria-label='Send prompt']": "Send button (aria)",
            # Assistant messages
            "article": "Article elements",
            "div[data-message-author-role='assistant']": "Assistant messages (data-role)",
            "[data-message-author-role='assistant']": "Assistant messages (any tag)",
            "div.agent-turn": "Agent turn divs",
            ".agent-turn": "Agent turn (any)",
            # Copy button
            "button[data-testid='copy-turn-action-button']": "Copy button (testid)",
            "button[aria-label='Copy']": "Copy button (aria)",
            # Stop button
            "button[aria-label='Stop generating']": "Stop button (aria)",
            "button[data-testid='stop-button']": "Stop button (testid)",
            # Login
            "button[data-testid='login-button']": "Login button",
            "button:has-text('Log in')": "Login text button",
        }

        for selector, desc in checks.items():
            try:
                elements = await page.query_selector_all(selector)
                count = len(elements)
                status = f"✅ Found {count}" if count > 0 else "❌ Not found"
                print(f"  {status} | {desc}: {selector}")
            except Exception as e:
                print(f"  ⚠️ Error  | {desc}: {selector} → {e}")

        # 3. Dump all article elements and their structure
        print("\n--- Article structure ---")
        article_info = await page.evaluate("""
            () => {
                const articles = document.querySelectorAll('article');
                const results = [];
                for (let i = 0; i < articles.length; i++) {
                    const a = articles[i];
                    results.push({
                        index: i,
                        id: a.id,
                        className: a.className,
                        attributes: Array.from(a.attributes).map(attr => `${attr.name}=${attr.value}`),
                        childTagSummary: Array.from(a.children).map(c => c.tagName + (c.className ? '.' + c.className.split(' ').join('.') : '')).slice(0, 5),
                        innerTextSnippet: (a.innerText || '').substring(0, 200),
                    });
                }
                return results;
            }
        """)
        if article_info:
            for a in article_info:
                print(f"\n  Article[{a['index']}]:")
                print(f"    id={a['id']}, class={a['className']}")
                print(f"    attrs: {a['attributes']}")
                print(f"    children: {a['childTagSummary']}")
                print(f"    text: {a['innerTextSnippet'][:100]}...")
        else:
            print("  No article elements found!")

        # 4. Look for any buttons with useful attributes
        print("\n--- All buttons with data-testid ---")
        buttons = await page.evaluate("""
            () => {
                const btns = document.querySelectorAll('button[data-testid]');
                return Array.from(btns).map(b => ({
                    testid: b.getAttribute('data-testid'),
                    ariaLabel: b.getAttribute('aria-label'),
                    textContent: (b.textContent || '').trim().substring(0, 80),
                    className: b.className.substring(0, 100),
                }));
            }
        """)
        for b in buttons:
            print(f"  testid={b['testid']}, aria={b['ariaLabel']}, text={b['textContent'][:50]}")

        # 5. Look for any copy-like buttons
        print("\n--- Buttons with 'copy' in any attribute ---")
        copy_buttons = await page.evaluate("""
            () => {
                const allBtns = document.querySelectorAll('button');
                const matches = [];
                for (const btn of allBtns) {
                    const attrs = Array.from(btn.attributes).map(a => a.value.toLowerCase()).join(' ');
                    const text = (btn.textContent || '').toLowerCase();
                    if (attrs.includes('copy') || text.includes('copy')) {
                        matches.push({
                            tagName: btn.tagName,
                            id: btn.id,
                            className: btn.className.substring(0, 100),
                            ariaLabel: btn.getAttribute('aria-label'),
                            dataTestid: btn.getAttribute('data-testid'),
                            text: (btn.textContent || '').trim().substring(0, 80),
                            allAttrs: Array.from(btn.attributes).map(a => `${a.name}="${a.value}"`).join(', '),
                        });
                    }
                }
                return matches;
            }
        """)
        for b in copy_buttons:
            print(f"  attrs: {b['allAttrs']}")
            print(f"  text: {b['text']}")
            print()

        # 6. Look for message role attributes anywhere
        print("\n--- Elements with data-message-* attributes ---")
        msg_elements = await page.evaluate("""
            () => {
                const all = document.querySelectorAll('[data-message-author-role], [data-message-id]');
                return Array.from(all).map(el => ({
                    tag: el.tagName,
                    role: el.getAttribute('data-message-author-role'),
                    messageId: el.getAttribute('data-message-id'),
                    className: el.className.substring(0, 100),
                    parentTag: el.parentElement?.tagName,
                }));
            }
        """)
        for m in msg_elements:
            print(f"  <{m['tag']}> role={m['role']}, msgId={m['messageId']}, class={m['className']}")

        # 7. Check for the new ChatGPT DOM patterns
        print("\n--- Checking newer ChatGPT DOM patterns ---")
        new_checks = {
            "[data-message-author-role]": "Any element with message role",
            "div[data-message-author-role='assistant'] button": "Buttons inside assistant messages",
            "main": "Main content area",
            "main article": "Articles in main",
            "[class*='markdown']": "Markdown class elements",
            "[class*='prose']": "Prose class elements",
            "[class*='result-streaming']": "Streaming indicator",
            "[class*='agent']": "Agent-related classes",
            "div[class*='group']": "Group divs",
            "button svg": "Buttons with SVGs",
            "[role='presentation']": "Presentation role elements",
            "div[data-testid]": "Divs with data-testid",
        }
        for selector, desc in new_checks.items():
            try:
                elements = await page.query_selector_all(selector)
                count = len(elements)
                if count > 0:
                    print(f"  ✅ {count:3d} | {desc}: {selector}")
            except Exception as e:
                print(f"  ⚠️ Error  | {desc}: {e}")

        # 8. Dump all data-testid values on the page
        print("\n--- All data-testid values on page ---")
        testids = await page.evaluate("""
            () => {
                const all = document.querySelectorAll('[data-testid]');
                return Array.from(all).map(el => ({
                    tag: el.tagName,
                    testid: el.getAttribute('data-testid'),
                    role: el.getAttribute('role'),
                }));
            }
        """)
        seen = set()
        for t in testids:
            key = f"{t['tag']}:{t['testid']}"
            if key not in seen:
                seen.add(key)
                print(f"  <{t['tag']}> data-testid=\"{t['testid']}\"")

        print("\n=== Diagnostic complete ===")
        print("\nPress ENTER to close browser...")
        input()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
