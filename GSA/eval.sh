method=$1
if [[ ${method} == 'gsa' ]]; then
    CUDA_VISIBLE_DEVICES=0 python gsa.py
else
    echo 'unknown argment for method'
fi
