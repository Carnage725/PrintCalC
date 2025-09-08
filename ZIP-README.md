# PrintCalC — Quick Start (for print shops)

**What is it?**  
PrintCalC estimates fair print prices by measuring each PDF page locally on your computer.  
No internet needed. Your files never leave your machine.

---

## Requirements
- Windows 10/11 or macOS 12+  
- Any modern browser (Chrome/Edge/Brave/Safari)

---

## How to run

### Windows
1) Unzip this file anywhere (e.g., Desktop).  
2) Open the folder **PrintCalC**.  
3) Double-click **PrintCalC.exe**.  
   - If you see *“Windows protected your PC”* → click **More info → Run anyway**.  
4) Your browser opens **PrintCalC** at `http://127.0.0.1:8501`.

### macOS
1) Unzip this file anywhere (e.g., Desktop).  
2) Open the folder **PrintCalC**.  
3) Right-click **PrintCalC** → **Open** → **Open** (first time only, Gatekeeper).  
4) Your browser opens **PrintCalC** at `http://127.0.0.1:8501`.

> If the browser doesn’t open automatically, manually enter the address shown above.

---

## Using the app (2 minutes)
1) **Choose printer** (left side): **Color** or **B/W**, and **Single-sided** or **Duplex**.  
2) **Pricing tabs** (top): set your slab prices for **Single-sided** and **Duplex**.  
3) **Analyze** tab: **Drag & drop a PDF** → see pages, sheets, and total ₹.  
   - **Blank pages** are ₹0.  
   - **Color printer:** even grey-only pages use the **lowest Color slab**.  
   - **B/W printer:** priced by **non-white %**.

**Tip:** For speed, use **Fast (200 DPI)**. Use **Accurate (300 DPI)** for thin details.

---

## Privacy
- Runs completely **offline** on your computer.  
- Your PDFs are **never uploaded**.

---

## Update / reinstall
- To update, download the new zip and **replace the entire folder**.  
- No installer, no admin rights needed.

---

## Troubleshooting
- **Security prompt:**  
  - Windows: *More info → Run anyway*  
  - macOS: Right-click → **Open** (first run only)
- **Firewall pop-up:** allow access; the app only uses `127.0.0.1` (your computer).
- **Nothing happens / blank page:** open `http://127.0.0.1:8501` manually.  
- **Large PDF is slow:** switch to **Fast** mode; set a small **Ignore border (mm)**.  
- **Quit the app:** close the black/terminal window that opened with it.

---

## License & source
- Licensed under **AGPL-3.0**.  
- Source & updates: https://github.com/Carnage725/PrintCalC
