# Offline Baseline

## Tacos
./tools/dist_train.sh configs/vslnet/benchmark/tacos_vslnet_h128_k7l4_lam5_seg64_inter32_neg07_b256.py 1 --validate --test-best --seed 2333 --deterministic
./tools/dist_train.sh configs/seqpan/dev/tacos_seqpan_h128_k7l4_sgpa2_seg64_inter32_neg07_b256.py 1 --validate --test-best --seed 2333 --deterministic
./tools/dist_train.sh configs/2dtan/benchmark/tacos_2dtan_n128_k5l8_seg64_win32_neg07_b32.py 1 --validate --test-best --seed 2333 --deterministic


## Anet
./tools/dist_train.sh configs/vslnet/benchmark/anet_vslnet_h128_k7l4_lam5_seg64_inter32_neg07_b256.py 1 --validate --test-best --seed 2333 --deterministic
./tools/dist_train.sh configs/seqpan/dev/anet_seqpan_h128_k7l4_sgpa2_seg64_inter32_neg07_b256.py 1 --validate --test-best --seed 2333 --deterministic
./tools/dist_train.sh configs/2dtan/benchmark/anet_2dtan_n64_k9l4_seg64_win32_neg07_b32.py 1 --validate --test-best --seed 2334 --deterministic
./tools/dist_train.sh configs/2dtan/benchmark/anet_2dtan_n64_k9l4_seg64_win32_neg07_b32.py 1 --validate --test-best --seed 2334 --deterministic

## MAD
./tools/dist_train.sh configs/2dtan/benchmark/mad_2dtan_n64_k7l4_seg32_win16_neg07_b128.py 1 --validate --test-best --seed 2333 --deterministic
./tools/dist_train.sh configs/vslnet/benchmark/mad_vslnet_h128_k7l4_lam5_seg32_inter16_neg07_b64.py 1 --validate --test-best --seed 2333 --deterministic
./tools/dist_train.sh configs/seqpan/dev/mad_seqpan_h128_k7l4_sgpa2_seg32_inter16_neg07_b64.py 1 --validate --test-best --seed 2333 --deterministic