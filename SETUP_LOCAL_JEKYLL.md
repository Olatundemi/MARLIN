# How to Run MARLIN GitHub Pages Locally on Windows

## Step 1: Install Ruby

### Option A: Using RubyInstaller (Recommended)

1. **Download Ruby Installer for Windows**:
   - Go to: https://rubyinstaller.org/downloads/
   - Download **Ruby+Devkit 3.2.2** (or latest 3.x)
   - The file should be named something like `rubyinstaller-3.2.2-1-x64.exe`

2. **Run the Installer**:
   - Double-click the downloaded `.exe` file
   - Accept the license agreement
   - **IMPORTANT**: Check the box **"Add Ruby executables to your PATH"**
   - Click **Install**
   - When prompted to install MSYS2 tools, select **Option 3** (recommended)

3. **Verify Installation**:
   ```bash
   ruby --version
   gem --version
   ```
   Both should return version numbers.

### Option B: Using Windows Package Manager (winget)

```bash
winget install RubyInstallerTeam.Ruby.3.2
```

---

## Step 2: Install Bundler

Open PowerShell as Administrator and run:

```bash
gem install bundler
bundler --version
```

---

## Step 3: Run Jekyll Locally

Navigate to your docs folder:

```bash
cd c:\Users\oibrahim\Documents\MARLIN\docs
```

Install dependencies:

```bash
bundle install
```

Start the Jekyll development server:

```bash
bundle exec jekyll serve
```

You should see output like:
```
Configuration file: c:/Users/oibrahim/Documents/MARLIN/docs/_config.yml
            Source: c:/Users/oibrahim/Documents/MARLIN/docs
       Destination: c:/Users/oibrahim/Documents/MARLIN/docs/_site
 Incremental build: enabled
      Generating... 
       Jekyll Feed: Generating feed for posts
                    done in 0.123 seconds.
 Auto-regeneration: enabled for 'c:/Users/oibrahim/Documents/MARLIN/docs'
    Server address: http://127.0.0.1:4000
  Server running...
  Press Ctrl-C to stop.
```

---

## Step 4: View Your Site

1. **Open your browser**
2. **Navigate to**: http://localhost:4000
3. Your MARLIN documentation will be displayed locally!

---

## Troubleshooting

### "bundle: command not found"
- Ruby/bundler not in PATH
- Solution: Restart PowerShell or reinstall Ruby ensuring "Add to PATH" is checked

### "Gem files' source is restricted to the user home directory"
```bash
bundle install --system
```

### Port 4000 Already in Use
```bash
bundle exec jekyll serve --port 4001
```
Then visit: http://localhost:4001

### SSL Certificate Error
```bash
bundle config set --global ignore_messages true
```

---

## Quick Copy-Paste Commands

**Full setup from scratch**:
```bash
cd c:\Users\oibrahim\Documents\MARLIN\docs
bundle install
bundle exec jekyll serve
```

Then open: http://localhost:4000

---

## What Gets Generated

When Jekyll runs, it creates a `_site` folder with:
- Static HTML files
- CSS files
- Everything needed to view your site locally
- This is what GitHub Pages deploys

**Note**: Don't commit the `_site` folder to git (it's in `.gitignore`)

---

## Making Changes

When running locally with Jekyll:
1. Edit any `.md` or `.scss` files in `docs/`
2. Jekyll **auto-regenerates** the site (you'll see "...done" messages)
3. Refresh your browser to see changes
4. **No need to restart** Jekyll for most changes

---

## Stopping the Server

Press **Ctrl + C** in the terminal to stop Jekyll.

---

Need help? Check:
- Ruby official: https://www.ruby-lang.org/
- Jekyll docs: https://jekyllrb.com/docs/
- GitHub Pages: https://docs.github.com/en/pages

