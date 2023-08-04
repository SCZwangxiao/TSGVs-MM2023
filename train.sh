# ActivityNet Captions and TACoS - Train & test
./tools/dist_train.sh configs/tpn/benchmark/anet_mtl_16_16_tpn_dec2_rnn_dot_s8_l64_b8*64_kd04.py 8 --validate --test-best
./tools/dist_train.sh configs/tpn/benchmark/tacos_mtl_16_tpn_dec1_rnn_dot_s8_l64_b8*64_kd02.py 8 --validate --test-best
# MAD dataset - Train
./tools/dist_train.sh configs/tpn/benchmark/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02.py 8 --validate
# MAD dataset - Test
for i in {0..19} # split the dataset into 20 chunks to avoid OOM
do
    ./tools/dist_test.sh work_dirs/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02.py \
    work_dirs/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02/best_R@1,IoU=0.3_epoch_13.pth 2 --eval R@N,IoU=M \
    --out work_dirs/mad_mtl_16_16_tpn_dec2_rnn_dot_s8_l32_b8*128_kd02/best_pred_${i}_20.pkl \
    --cfg-options data.workers_per_gpu=10 data.test.portion=[$i,20]
done