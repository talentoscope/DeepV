#!/bin/bash

#!/bin/bash

#!/bin/bash

echo "DeepV Dataset Download Script"
echo "=============================="
echo ""

# Golden Set (working)
echo "✅ Downloading Golden Set (evaluation dataset from paper)..."
FILEID=1dDs06LsLNQUg9HvUwNBIq-95bjmRAiMh
FILENAME=golden_set.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

# Extract golden_set.zip
if command -v unzip >/dev/null 2>&1; then
    unzip golden_set.zip
    rm golden_set.zip
    rm -rf __MACOSX
else
    echo "Using PowerShell Expand-Archive for golden_set..."
    powershell.exe -Command "Expand-Archive -Path golden_set.zip -DestinationPath . -Force; Remove-Item golden_set.zip"
fi

echo ""
echo "❌ REMOVED: ABC dataset (links broken, use synthetic generation instead)"
echo "❌ REMOVED: Background images (Google Drive link broken)"
echo "❌ REMOVED: Dataset_of_cleaning (requires manual Yandex Disk download)"
echo "❌ REMOVED: Precision Floorplan (website down)"
echo ""
echo "✅ SUCCESS: Golden Set downloaded and extracted"
echo ""
echo "For missing datasets, use synthetic generation:"
echo "python scripts/create_test_datasets.py"



# If doesn't work try this type of function
#
#function gdrive_download () {
#  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
#  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
#  rm -f /tmp/cookies.txt
#}
#
#gdrive_download 1HIvMOJqm77flpvJWNpLHSxOpm8et_s_W Background.tar.gz
#
#
#tar -zxvf Background.tar.gz
#rm Background.tar.gz
#mv Background ../data/Background
#rm -rf __MACOSX


