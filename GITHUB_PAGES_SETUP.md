# GitHub Pages Setup Guide

## Overview

Your MARLIN model development documentation is now ready for GitHub Pages publication. This guide walks you through enabling and publishing your docs site.

## What's Been Created

```
docs/
├── _config.yml                    # Jekyll configuration
├── Gemfile                         # Ruby dependencies
├── .gitignore                      # Git ignore rules
├── index.md                        # Home page
├── architecture.md                 # Model architecture details
├── methodology.md                  # Training methodology
├── experiments.md                  # Experiment configurations  
├── results.md                      # Results & performance
├── quick-reference.md              # Quick lookup guide
└── GUIDE.md                        # Documentation guide
```

## Step 1: Enable GitHub Pages

### On GitHub
1. Go to your repository: `https://github.com/Olatundemi/MARLIN`
2. Click **Settings** → **Pages**
3. Under "Build and deployment":
   - **Source**: Select "Deploy from a branch"
   - **Branch**: Select `main`
   - **Folder**: Select `/ (root)` → change to `/docs`
4. Click **Save**

*Your site should now be live at: `https://olatundemi.github.io/MARLIN/`*

### Alternative: Using GitHub Actions (Recommended for more control)

Create `.github/workflows/jekyll.yml`:

```yaml
name: Build and Deploy Jekyll

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.1'
          bundler-cache: true
          working-directory: ./docs
      
      - name: Setup Pages
        uses: actions/configure-pages@v4
      
      - name: Build with Jekyll
        run: bundle exec jekyll build --baseurl "${{ steps.pages.outputs.base_url }}"
        working-directory: ./docs
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./docs/_site
      
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v3
```

---

## Step 2: Test Locally (Optional but Recommended)

### Prerequisites
Ensure you have Ruby 2.7+ installed:
```bash
ruby --version
```

### Install & Run
```bash
cd docs/

# Install dependencies
bundle install

# Build and serve
bundle exec jekyll serve

# Open browser
# Navigate to http://localhost:4000
```

### Troubleshooting Local Build
```bash
# Check for errors
bundle exec jekyll build --verbose

# Check Jekyll doctor
bundle exec jekyll doctor

# Clear cache if issues persist
rm -rf _site/
rm -rf .jekyll-cache/
bundle exec jekyll serve
```

---

## Step 3: Commit and Push

```bash
# Stage Documentation files
git add docs/

# Commit
git commit -m "Add GitHub Pages documentation for MARLIN model development

- Index page with project overview
- Detailed architecture documentation
- Training methodology guide
- Experiment configurations
- Results tracking page
- Quick reference guide
- Jekyll configuration and setup"

# Push to main
git push origin main
```

---

## Step 4: Verify Deployment

1. **Check GitHub Actions** (if using workflows):
   - Go to repository → **Actions** tab
   - Look for "Build and Deploy Jekyll" workflow
   - Verify it passed ✓

2. **Visit Your Site**:
   - URL: `https://olatundemi.github.io/MARLIN/`
   - Should see the homepage with navigation menu

3. **Test Navigation**:
   - Click through: Home → Architecture → Methodology → Experiments → Results
   - All links should work

---

## Customization Options

### Change Theme
Edit `docs/_config.yml`:
```yaml
# Current theme
theme: jekyll-theme-minimal

# Other options:
# - jekyll-theme-slate
# - jekyll-theme-cayman
# - jekyll-theme-dinky
# - jekyll-theme-leap-day
# - jekyll-theme-merlot
```

### Add Your Logo
1. Upload logo image to `docs/assets/logo.png`
2. Edit `_config.yml`:
```yaml
logo: /assets/logo.png
```

### Customize Colors
Create `docs/assets/css/style.scss`:
```scss
---
---

@import "{{ site.theme }}";

// Your custom CSS here
a {
  color: #0066cc;
}
```

### Add Custom Domain
1. In GitHub Settings → Pages
2. Under "Custom domain", enter your domain
3. Follow DNS setup instructions for your registrar

---

## Documentation Workflow

### After Running an Experiment
1. Record results in `docs/results.md`
2. Update experiment status from `[Pending]` to `[Completed]`
3. Fill in performance metrics table
4. Add observations and model checkpoint path
5. Commit and push:
```bash
git add docs/results.md
git commit -m "Update results for Exp 1 baseline model"
git push origin main
```

### When Adding New Experiments
1. Add configuration to `docs/experiments.md`
2. Create corresponding results entry in `docs/results.md`
3. Update [Experiments](experiments.html) page

### Updating Architecture/Methodology
1. Edit `docs/architecture.md` or `docs/methodology.md`
2. Commit changes:
```bash
git commit -m "Update architecture documentation - add hybrid PE details"
git push origin main
```

---

## Navigation Structure

The `_config.yml` defines the main navigation. Current structure:

```yaml
nav:
  - name: Home
    url: /
  - name: Architecture
    url: /architecture
  - name: Methodology
    url: /methodology
  - name: Experiments
    url: /experiments
  - name: Results
    url: /results
```

To modify, edit the `nav` section in `_config.yml`.

---

## Markdown Tips

### Internal Links
```markdown
# Link to another page
[See Methodology](methodology.html)
[Quick Reference](quick-reference.html)

# Link to specific section
[Attention Mechanism](#attention-mechanism)
```

### Code Blocks
```markdown
\`\`\`python
def my_function():
    return "code highlighted"
\`\`\`

\`\`\`bash
$ echo "Commands highlighted"
\`\`\`
```

### Tables
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |
```

### Alerts (Rendered as blockquotes)
```markdown
> **Note**: This is important information.

> **Warning**: Be careful with this setting.
```

---

## Troubleshooting

### Site Not Building
```bash
# Check for errors
git push origin main  # trigger GitHub Actions

# View workflow logs in Actions tab
# Or check Jekyll doctor locally
bundle exec jekyll doctor --workingdir ./docs
```

### Links Broken on Published Site
- Use `.html` extension: `[link](page.html)` not `[link](page.md)`
- Use relative paths: `../page.html` not `/page.html`

### CSS/Images Not Showing
- Ensure files in `docs/assets/` directory
- Reference as: `/assets/filename.ext`
- Clear browser cache: Ctrl+Shift+Delete

### Markdown Not Rendering Properly
- Check for missing closing backticks
- Verify indentation (Jekyll uses 2 spaces)
- Use proper YAML frontmatter

---

## GitHub Pages Settings (Useful Links)

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Minimal Theme Guide](https://github.com/pages-themes/minimal)
- [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

---

## Quality Checklist

Before considering your docs "complete":

- [ ] All navigation links work
- [ ] No broken internal links
- [ ] Code examples syntax-highlighted properly
- [ ] Tables display correctly
- [ ] Images appear (if any added)
- [ ] Mobile friendly (test on phone)
- [ ] No typos or formatting issues
- [ ] Complete architecture documentation
- [ ] Methodology clearly explained
- [ ] Experiment templates provided
- [ ] Results template ready for data entry

---

## Next Steps

1. ✅ Documentation created
2. ⬜ Enable GitHub Pages in Settings
3. ⬜ Test locally with Jekyll
4. ⬜ Commit and push to main
5. ⬜ Verify site is live
6. ⬜ Run first experiment and populate results
7. ⬜ Share documentation URL with team

---

## Support

For issues with Jekyll:
```bash
bundle exec jekyll doctor --workingdir ./docs
```

For GitHub Pages issues:
- Check Actions tab for build logs
- Review GitHub Pages documentation
- Check repository Settings → Pages

---

**Setup completed: February 2026**

Your documentation is ready to showcase your MARLIN model development! 🚀
