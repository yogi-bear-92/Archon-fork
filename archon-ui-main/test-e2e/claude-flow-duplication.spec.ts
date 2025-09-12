import { test, expect } from '@playwright/test';

test.describe('Claude Flow Duplication Debug', () => {
  test('should investigate Claude Flow duplication', async ({ page }) => {
    console.log('🚀 Investigating Claude Flow component duplication...');
    
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Look for all Claude Flow headers
    const claudeFlowHeaders = page.locator('h2:has-text("Claude Flow Integration")');
    const headerCount = await claudeFlowHeaders.count();
    
    console.log(`\n🔍 Found ${headerCount} Claude Flow Integration headers`);
    
    // Examine each header
    for (let i = 0; i < headerCount; i++) {
      const header = claudeFlowHeaders.nth(i);
      const isVisible = await header.isVisible();
      const boundingBox = await header.boundingBox();
      const className = await header.getAttribute('class');
      
      console.log(`Header ${i + 1}:`);
      console.log(`  - Visible: ${isVisible}`);
      console.log(`  - Position: ${JSON.stringify(boundingBox)}`);  
      console.log(`  - Class: ${className}`);
    }
    
    // Look for all "About Claude Flow Integration" text
    const aboutSections = page.locator('text=About Claude Flow Integration');
    const aboutCount = await aboutSections.count();
    console.log(`\n📖 Found ${aboutCount} "About Claude Flow Integration" sections`);
    
    // Check for tab duplications
    const swarmTabs = page.locator('text=Swarm Control');
    const tabCount = await swarmTabs.count();
    console.log(`🔗 Found ${tabCount} "Swarm Control" elements`);
    
    // Look for any component content that's actually visible
    const healthStatus = page.locator('text=Service Healthy').first();
    const aboutSection = page.locator('text=About Claude Flow Integration').first();
    const sparcText = page.locator('text=SPARC methodology').first();
    
    console.log(`\n✅ Component Content Visibility:`);
    console.log(`  - Health status: ${await healthStatus.isVisible()}`);
    console.log(`  - About section: ${await aboutSection.isVisible()}`);
    console.log(`  - SPARC text: ${await sparcText.isVisible()}`);
    
    // Take screenshot
    await page.screenshot({ path: 'claude-flow-duplication-debug.png', fullPage: true });
    console.log('📸 Screenshot saved as claude-flow-duplication-debug.png');
    
    expect(true).toBe(true);
  });
});