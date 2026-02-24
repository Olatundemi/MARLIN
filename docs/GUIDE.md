# Documentation Guide

This directory contains the GitHub Pages documentation for the MARLIN model development project.

## Structure

```
docs/
├── _config.yml              # Jekyll configuration
├── README.md               # This file
├── index.md                # Main homepage
├── architecture.md         # Model architecture details
├── methodology.md          # Training methodology & workflow
├── experiments.md          # Experiment configurations
├── results.md              # Results and performance metrics
└── assets/                 # Images and diagrams (optional)
```

## Quick Navigation

- **[Home](index.md)** - Project overview and key features
- **[Architecture](architecture.md)** - Detailed component descriptions
- **[Methodology](methodology.md)** - Training procedures and best practices
- **[Experiments](experiments.md)** - Experimental setups and configurations
- **[Results](results.md)** - Performance results and analysis

## Building Locally

### Prerequisites
- Ruby >= 2.7
- Jekyll >= 3.9

### Setup
```bash
cd docs/
bundle install  # Install dependencies
```

### Run Locally
```bash
bundle exec jekyll serve
# Navigate to http://localhost:4000
```

## Publishing to GitHub Pages

### Option 1: Direct Push (Recommended)
1. Commit changes to `docs/` folder
2. Push to main branch
3. Enable GitHub Pages in repository settings:
   - **Source**: Deploy from a branch
   - **Branch**: main
   - **Folder**: /docs

### Option 2: GitHub Actions
Create `.github/workflows/jekyll-build.yml`:
```yaml
name: Build and Deploy Jekyll

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: 3.1
          bundler-cache: true
          working-directory: ./docs
      
      - name: Build Jekyll
        run: bundle exec jekyll build
        working-directory: ./docs
```

## Content Guidelines

### Adding New Pages
1. Create markdown file in `docs/` (e.g., `new-page.md`)
2. Add frontmatter:
```yaml
---
layout: default
title: Page Title
---

# Heading
Content here...
```

3. Update `_config.yml` navigation if needed

### File Naming
- Use lowercase with hyphens: `my-page.md`
- Keep filenames descriptive

### Markdown Formatting
- Use setext-style headers for main titles: `===`
- Use ATX-style for subsections: `##`, `###`
- Include code blocks with language specification
- Use tables for structured data

## Key Sections to Update

### After Each Experiment
1. Update [Results](results.md) with metrics
2. Add checkpoint link in results table
3. Note key observations in findings section

### Before Code Changes
1. Update [Architecture](architecture.md) if model changes
2. Update [Methodology](methodology.md) if training changes
3. Add new [Experiments](experiments.md) entry if needed

## Theme Customization

Using Jekyll Minimal theme. Customize by:
1. Create `_layouts/default.html` (optional)
2. Create `assets/css/style.scss` for CSS overrides
3. Update color variables in `_config.yml`

## Troubleshooting

### Site not building
```bash
# Check for Jekyll errors
bundle exec jekyll build --verbose

# Validate markdown
bundle exec jekyll doctor
```

### Links not working
- Use relative paths: `[text](../path/file.md)`
- Avoid leading slashes for relative links
- Use `.html` extension for links (Jekyll converts `.md` → `.html`)

### Images not showing
Place images in `docs/assets/` and reference as:
```markdown
![alt text](/assets/image.png)
```

## Deployment Checklist

Before making docs public:
- [ ] All pages have proper titles and metadata
- [ ] Links are working (test locally first)
- [ ] Code examples are properly formatted
- [ ] Tables are complete and accurate
- [ ] No sensitive information in documentation
- [ ] Repository is set to public

## Additional Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Markdown Syntax](https://www.markdownguide.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Minimal Theme Documentation](https://github.com/pages-themes/minimal)

---

**Documentation maintained for**: MARLIN - Multi-Scale Temporal Encoding for Epidemiological Prediction

Last updated: February 2026
