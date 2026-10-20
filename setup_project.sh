#!/usr/bin/env bash
# Run once:  bash setup_project.sh
set -e

PROJECT=badminton_analysis
mkdir -p $PROJECT/{raw_videos,clips,keypoints,features,labels,models,results/{plots,annotated_videos},notebooks,src}
cd $PROJECT

cat > requirements.txt << 'EOF'
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
opencv-python
ultralytics
torch
tqdm
yt-dlp
streamlit
EOF

cat > labels/clips_template.csv << 'EOF'
clip_id,source_video,start_sec,end_sec,player,shot_type,split
EOF

cat > .gitignore << 'EOF'
venv/
raw_videos/
clips/
keypoints/
*.pt
__pycache__/
EOF

python3 -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate with: source $PROJECT/venv/bin/activate  (Windows: venv\\Scripts\\activate)"
