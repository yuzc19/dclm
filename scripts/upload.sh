# O=gs://cmu-gpucloud-zichunyu
# S=/mnt/localssd/zichunyu
# mkdir -p $S
# mkdir -p $D
# #copy the data to the local storage
# gcloud storage rsync $O/checkpoints $S/checkpoints --recursive
# # run your stuff
# # copy your stuff back to the object storage
# gcloud storage rsync $S/test-run $O --recursive
# # clean up local storage when done
# rm -fr $S

names=$(ls /tmp/dclm_logs)

for name in $names; do
    echo "Uploading $name"

    save_file="/tmp/dclm_logs/$name"
    save_link="gs://cmu-gpucloud-zichunyu/out/dclm_logs/data-limited-pretraining/$name"

    gcloud storage rsync $save_file $save_link --recursive > cp.log 2>&1
done