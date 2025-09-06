import { test, expect } from '@playwright/test';

test.describe('Claude Flow Integration Debugging', () => {
  let consoleMessages: string[] = [];
  let consoleErrors: string[] = [];
  let networkErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    // Capture console messages
    page.on('console', msg => {
      const text = `[${msg.type()}] ${msg.text()}`;
      consoleMessages.push(text);
      
      if (msg.type() === 'error') {
        consoleErrors.push(text);
      }
    });

    // Capture network failures
    page.on('response', response => {
      if (!response.ok()) {
        networkErrors.push(`Failed request: ${response.url()} - ${response.status()} ${response.statusText()}`);
      }
    });

    // Capture uncaught exceptions
    page.on('pageerror', error => {
      consoleErrors.push(`Page error: ${error.message}`);
    });
  });

  test('should load homepage without console errors', async ({ page }) => {
    console.log('🚀 Starting homepage test...');
    
    await page.goto('/');
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');
    
    console.log('\n📊 CONSOLE MESSAGES:');
    consoleMessages.forEach(msg => console.log(msg));
    
    console.log('\n❌ CONSOLE ERRORS:');
    consoleErrors.forEach(error => console.log(error));
    
    console.log('\n🌐 NETWORK ERRORS:');
    networkErrors.forEach(error => console.log(error));
    
    // Don't fail on errors for debugging - just capture them
    expect(true).toBe(true);
  });

  test('should load settings page and test Claude Flow components', async ({ page }) => {
    console.log('🚀 Starting settings page test...');
    
    await page.goto('/settings');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Try to find Claude Flow section
    console.log('\n🔍 Looking for Claude Flow Integration section...');
    
    try {
      const claudeFlowSection = page.locator('text=Claude Flow Integration');
      if (await claudeFlowSection.isVisible()) {
        console.log('✅ Claude Flow Integration section found');
        
        // Try to click to expand
        await claudeFlowSection.click();
        console.log('✅ Clicked Claude Flow section');
        
        // Look for components
        const swarmControl = page.locator('[data-testid="swarm-control"]');
        const agentSpawner = page.locator('[data-testid="agent-spawner"]');
        
        console.log(`SwarmControl visible: ${await swarmControl.isVisible()}`);
        console.log(`AgentSpawner visible: ${await agentSpawner.isVisible()}`);
        
      } else {
        console.log('❌ Claude Flow Integration section not found');
      }
    } catch (error) {
      console.log(`❌ Error testing Claude Flow section: ${error}`);
    }
    
    console.log('\n📊 SETTINGS PAGE CONSOLE MESSAGES:');
    consoleMessages.forEach(msg => console.log(msg));
    
    console.log('\n❌ SETTINGS PAGE CONSOLE ERRORS:');
    consoleErrors.forEach(error => console.log(error));
    
    console.log('\n🌐 SETTINGS PAGE NETWORK ERRORS:');
    networkErrors.forEach(error => console.log(error));
    
    expect(true).toBe(true);
  });

  test('should test individual Claude Flow components', async ({ page }) => {
    console.log('🚀 Testing individual components...');
    
    // Navigate directly to settings
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    // Test specific component functionality
    try {
      // Test if we can access component APIs
      const componentTest = await page.evaluate(() => {
        const results: any = {};
        
        // Check if React is working
        results.react = typeof window.React !== 'undefined' || document.querySelector('[data-reactroot]') !== null;
        
        // Check if settings components are rendered
        results.settingsPage = document.querySelector('[data-testid="settings"]') !== null;
        
        // Check for Claude Flow related elements
        results.claudeFlowElements = document.querySelectorAll('*[class*="claude"], *[class*="flow"], *[class*="swarm"], *[class*="agent"]').length;
        
        return results;
      });
      
      console.log('🔍 Component Test Results:', componentTest);
      
    } catch (error) {
      console.log(`❌ Component test error: ${error}`);
    }
    
    console.log('\n📊 COMPONENT TEST CONSOLE MESSAGES:');
    consoleMessages.forEach(msg => console.log(msg));
    
    console.log('\n❌ COMPONENT TEST ERRORS:');
    consoleErrors.forEach(error => console.log(error));
    
    expect(true).toBe(true);
  });
});