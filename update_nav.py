import os

# The dynamic navbar placeholder and the fetching script
NAV_PLACEHOLDER = '<div id="nav-placeholder"></div>'
FETCH_SCRIPT = """
<script>
fetch('/navbar.html')
    .then(response => response.text())
    .then(data => {
        document.getElementById('nav-placeholder').innerHTML = data;
    });
</script>
"""

# Count updated files
updated_count = 0

print("Scanning folders and updating HTML files...")

# Walk through all directories and subdirectories starting from the root folder
for root, dirs, files in os.walk("."):
    for file in files:
        # Target HTML files, but skip navbar.html itself
        if file.endswith(".html") and file != "navbar.html":
            file_path = os.path.join(root, file)
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            modified = False
            
            # 1. Check if the nav-placeholder is missing; if so, inject it into the header
            if "nav-placeholder" not in content:
                if "<header>" in content:
                    content = content.replace("<header>", "<header>\n    " + NAV_PLACEHOLDER)
                    modified = True
                elif "<body>" in content:
                    # If there's no header tag, wrap it near the top of the body
                    content = content.replace("<body>", "<body>\n<header>\n    " + NAV_PLACEHOLDER + "\n</header>")
                    modified = True

            # 2. Check if the fetch script is missing; if so, append it before </body>
            if "fetch('/navbar.html')" not in content:
                if "</body>" in content:
                    content = content.replace("</body>", FETCH_SCRIPT + "\n</body>")
                    modified = True

            # Save the file only if changes were made
            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✔ Updated: {file_path}")
                updated_count += 1

print(f"\nBatch update complete! Total files updated: {updated_count}")