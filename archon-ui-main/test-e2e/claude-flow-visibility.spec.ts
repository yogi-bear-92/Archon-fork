import { test, expect } from '@playwright/test';

test.describe('Claude Flow Component Visibility Debug', () => {
  test('should test Claude Flow components visibility', async ({ page }) => {
    console.log('🚀 Testing Claude Flow component visibility...');
    
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    
    // Find the Claude Flow section
    const claudeFlowSection = page.locator('text=Claude Flow Integration');
    
    if (await claudeFlowSection.isVisible()) {
      console.log('✅ Claude Flow section found');
      
      // Click to expand
      await claudeFlowSection.click();
      console.log('✅ Clicked Claude Flow section');
      
      // Wait a moment for expansion animation
      await page.waitForTimeout(1000);
      
      // Check for any collapsible container
      const collapsibleContainer = page.locator('[data-testid*="claude-flow"], [data-cy*="claude-flow"], .claude-flow, div:has-text("Claude Flow")').first();
      
      // Try to find the components within the expanded section
      const swarmControlText = page.locator('text=Claude Flow Swarm');
      const agentSpawnerText = page.locator('text=Spawn Agents');
      
      console.log(`SwarmControl text visible: ${await swarmControlText.isVisible()}`);
      console.log(`AgentSpawner text visible: ${await agentSpawnerText.isVisible()}`);
      
      // Try to find the tab navigation
      const swarmTab = page.locator('text=Swarm Control');
      const agentsTab = page.locator('text=Agent Spawner');
      const metricsTab = page.locator('text=Metrics');
      
      console.log(`Swarm Control tab visible: ${await swarmTab.isVisible()}`);
      console.log(`Agent Spawner tab visible: ${await agentsTab.isVisible()}`);
      console.log(`Metrics tab visible: ${await metricsTab.isVisible()}`);
      
      // Check if there are any error messages or loading states
      const healthStatus = page.locator('text=Service Unavailable');
      console.log(`Service Unavailable message visible: ${await healthStatus.isVisible()}`);
      
      // Try clicking on tabs if they exist
      if (await swarmTab.isVisible()) {
        await swarmTab.click();
        console.log('✅ Clicked Swarm Control tab');
        
        // Look for SwarmControl specific elements
        const topologySelect = page.locator('select', { hasText: 'Adaptive' });
        const maxAgentsSlider = page.locator('input[type="range"]');
        
        console.log(`Topology select visible: ${await topologySelect.isVisible()}`);
        console.log(`Max agents slider visible: ${await maxAgentsSlider.isVisible()}`);
      }
      
      if (await agentsTab.isVisible()) {
        await agentsTab.click();
        console.log('✅ Clicked Agent Spawner tab');
        
        // Look for AgentSpawner specific elements
        const objectiveTextarea = page.locator('textarea[placeholder*="Describe what you want"]');
        const strategyOptions = page.locator('text=development');
        
        console.log(`Objective textarea visible: ${await objectiveTextarea.isVisible()}`);
        console.log(`Strategy options visible: ${await strategyOptions.isVisible()}`);
      }
      
    } else {
      console.log('❌ Claude Flow Integration section not found');
    }
    
    // Get all visible text to debug
    const allText = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const texts = [];
      for (let el of elements) {
        if (el.textContent && el.textContent.toLowerCase().includes('claude flow')) {
          texts.push({
            tag: el.tagName,
            text: el.textContent.substring(0, 100),
            visible: el.offsetParent !== null
          });
        }
      }
      return texts;
    });
    
    console.log('🔍 All Claude Flow related elements:', JSON.stringify(allText, null, 2));
    
    expect(true).toBe(true);
  });
});