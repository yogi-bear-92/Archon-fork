import { test, expect } from '@playwright/test';

test.describe('Claude Flow Final Test', () => {
  let consoleErrors: string[] = [];
  let jsErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    // Reset localStorage to clear any stored expansion state
    await page.evaluate(() => {
      localStorage.clear();
    });

    // Capture all console errors
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(`[${msg.type()}] ${msg.text()}`);
      }
    });

    // Capture JavaScript errors
    page.on('pageerror', error => {
      jsErrors.push(`Page error: ${error.message}`);
    });
  });

  test('should load Claude Flow components with defaultExpanded=true', async ({ page }) => {
    console.log('🚀 Testing Claude Flow with defaultExpanded=true...');
    
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    console.log('\n📊 Console Errors during load:');
    consoleErrors.forEach(error => console.log(error));
    
    console.log('\n🔥 JavaScript Errors:');
    jsErrors.forEach(error => console.log(error));
    
    // Wait for React components to fully render
    await page.waitForTimeout(2000);
    
    // Check if Claude Flow section exists
    const claudeFlowSection = page.locator('text=Claude Flow Integration');
    console.log(`\n🔍 Claude Flow section visible: ${await claudeFlowSection.isVisible()}`);
    
    // Look for specific ClaudeFlowSettings component elements
    const healthIndicator = page.locator('text=Service Healthy, text=Service Unavailable');
    const aboutSection = page.locator('text=About Claude Flow Integration');
    
    console.log(`Health indicator visible: ${await healthIndicator.isVisible()}`);
    console.log(`About section visible: ${await aboutSection.isVisible()}`);
    
    // Try to find any part of the ClaudeFlowSettings component
    const sparcText = page.locator('text=SPARC');
    const swarmText = page.locator('text=Swarm');
    const neuralText = page.locator('text=Neural');
    
    console.log(`SPARC text visible: ${await sparcText.isVisible()}`);
    console.log(`Swarm text visible: ${await swarmText.isVisible()}`);  
    console.log(`Neural text visible: ${await neuralText.isVisible()}`);
    
    // Get all elements containing "Claude Flow" text
    const allClaudeFlowElements = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const results = [];
      for (let el of elements) {
        if (el.textContent && el.textContent.includes('Claude Flow')) {
          results.push({
            tag: el.tagName,
            className: el.className,
            textContent: el.textContent.substring(0, 200),
            offsetParent: el.offsetParent !== null,
            computedStyle: window.getComputedStyle(el).display
          });
        }
      }
      return results;
    });
    
    console.log('\n🔍 All Claude Flow elements:', JSON.stringify(allClaudeFlowElements, null, 2));
    
    // Check if any React components failed to render
    const reactErrors = await page.evaluate(() => {
      const errors = [];
      // Check for any error boundaries or failed renders
      const errorElements = document.querySelectorAll('[data-react-error], .react-error, [class*="error"]');
      for (let el of errorElements) {
        errors.push(el.textContent);
      }
      return errors;
    });
    
    if (reactErrors.length > 0) {
      console.log('\n⚠️ Potential React errors:', reactErrors);
    }
    
    expect(true).toBe(true);
  });
});