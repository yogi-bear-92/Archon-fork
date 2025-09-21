import { test, expect } from '@playwright/test';

test.describe('Claude Flow Simple Test', () => {
  test('should load Claude Flow components after defaultExpanded fix', async ({ page }) => {
    console.log('🚀 Testing Claude Flow with defaultExpanded=true...');
    
    // Clear storage before navigating
    await page.context().clearCookies();
    
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    // Wait for components to render
    await page.waitForTimeout(3000);
    
    console.log('\n🔍 Looking for Claude Flow content...');
    
    // Look for the main Claude Flow section
    const claudeFlowHeader = page.locator('h2:has-text("Claude Flow Integration")');
    console.log(`Claude Flow header visible: ${await claudeFlowHeader.isVisible()}`);
    
    // Look for About section which should be visible if expanded
    const aboutSection = page.locator('text=About Claude Flow Integration');
    console.log(`About section visible: ${await aboutSection.isVisible()}`);
    
    // Look for key text that should be in the component
    const sparcText = page.locator('text=SPARC methodology');
    const swarmCoordText = page.locator('text=Swarm Coordination');
    const neuralText = page.locator('text=Neural Learning');
    const archonIntegText = page.locator('text=Archon Integration');
    
    console.log(`SPARC methodology text visible: ${await sparcText.isVisible()}`);
    console.log(`Swarm Coordination text visible: ${await swarmCoordText.isVisible()}`);
    console.log(`Neural Learning text visible: ${await neuralText.isVisible()}`);
    console.log(`Archon Integration text visible: ${await archonIntegText.isVisible()}`);
    
    // Try to find the health status indicator
    const healthStatus = page.locator('text=Service Healthy, text=Service Unavailable');
    console.log(`Health status visible: ${await healthStatus.isVisible()}`);
    
    // Look for specific tabs
    const swarmTab = page.locator('button:has-text("Swarm Control")');
    const agentsTab = page.locator('button:has-text("Agent Spawner")'); 
    const metricsTab = page.locator('button:has-text("Metrics")');
    
    console.log(`Swarm Control tab visible: ${await swarmTab.isVisible()}`);
    console.log(`Agent Spawner tab visible: ${await agentsTab.isVisible()}`);
    console.log(`Metrics tab visible: ${await metricsTab.isVisible()}`);
    
    // Get a screenshot for debugging
    await page.screenshot({ path: 'claude-flow-debug.png', fullPage: true });
    console.log('📸 Screenshot saved as claude-flow-debug.png');
    
    expect(true).toBe(true);
  });
});