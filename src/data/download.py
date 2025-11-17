import os
import requests

BASE_REPO = "https://raw.githubusercontent.com/NTNU-HPC-Lab/LS-CAT/master"
KERNEL_ROOT = "data/kernels"
OUTPUT_DIR = "./downloaded_kernels"


def download_kernel(major_id, minor_id, filename):
    """
    Download one kernel:
    kernels/<major_id>/<minor_id>/<filename>
    """
    # Build path inside the repo
    rel_path = f"{KERNEL_ROOT}/{major_id}/{minor_id}/{filename}"
    url = f"{BASE_REPO}/{rel_path}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{major_id}_{minor_id}_{filename}")

    print(f"Downloading {major_id}, {minor_id}, {filename}")

    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {url} (status {r.status_code})")

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


# (major_id, minor_id, filename)
KERNELS_TO_DOWNLOAD = [
    (1, 14, "getColNorms.cu"),
    (1, 15, "gpuFindMax.cu"),
    (1, 16, "gpuSwapCol.cu"),
    (1, 17, "makeHVector.cu"),
    (1, 18, "UpdateHHNorms.cu"),
    (1, 19, "gpuKendall.cu"),
    (1, 20, "getRestricted.cu"),
    (1, 21, "getUnrestricted.cu"),
    (1, 22, "ftest.cu"),
    (1, 36, "gpuMeans.cu"),
    (1, 37, "gpuSD.cu"),
    (1, 38, "gpuPMCC.cu"),
    (1, 39, "gpuMeansNoTest.cu"),
    (1, 40, "gpuSDNoTest.cu"),
    (1, 41, "gpuPMCCNoTest.cu"),
    (1, 42, "gpuSignif.cu"),
    (1, 43, "dUpdateSignif.cu"),
    (1, 44, "noNAsPmccMeans.cu"),
    (1, 45, "scale.cu"),
    (1, 46, "get_bin_scores.cu"),
    (1, 47, "get_entropy.cu"),
    (1, 48, "get_mi.cu"),
    (10383, 24, "hello.cu"),
    (10383, 28, "hello.cu"),
    (104, 0, "total.cu"),
    (1040, 1, "delete_rows_and_columns.cu"),
    (10406, 0, "kernelAddConstant.cu"),
    (1000, 5, "kernel.cu"),
    (10024, 0, "cuda_cmp_kernel.cu"),
    (10047, 0, "add.cu"),
    (10047, 1, "add.cu"),
    (1008, 0, "blob_rearrange_kernel2_1d.cu"),
    (1008, 7, "blob_rearrange_kernel2.cu"),
    (10121, 1, "addKernel.cu"),
    (1001, 0, "grayscaleVer2D.cu"),
    (10121, 0, "MedianFilter_gpu.cu"),
    (1029, 0, "gpu_grayscale.cu"),
    (1029, 1, "gpu_gaussian.cu"),
    (1029, 2, "gpu_sobel.cu"),
    (10383, 9, "remove_redness_from_coordinates.cu"),
    (10393, 4, "relu.cu"),
    (10393, 5, "relu_backward.cu"),
    (10615, 4, "tonemap_adaptive.cu"),
    (10615, 5, "tonemap_logarithmic.cu"),
    (10383, 20, "global_reduce_kernel.cu"),
    (10383, 21, "shmem_reduce_kernel.cu"),
    (104, 5, "scan.cu"),
    (104, 4, "post_scan.cu"),
    (10492, 1, "multiplyElementKernel.cu"),
    (10121, 3, "reduce.cu"),
    (10121, 5, "reduce.cu"),
    (10223, 26, "phglobal_reduce_kernel.cu"),
    (10802, 3, "Histogram_kernel.cu"),
    (10802, 7, "non_max_supp_kernel.cu"),
    (10802, 8, "hyst_kernel.cu"),
    (11022, 3, "vsub.cu"),
    (11022, 4, "vmul.cu"),
    (11022, 5, "vdiv.cu"),
    (11022, 6, "dummy.cu"),
    (11846, 10, "atbashGPU.cu"),
    (11647, 0, "add.cu"),
    (11647, 1, "add.cu"),
    (11647, 2, "add.cu"),
    (11647, 3, "add.cu"),
    (11647, 4, "add.cu"),
    (11647, 10, "add.cu"),
    (1000, 24, "transposedMatrixKernel.cu"),
    (1000, 25, "additionMatricesKernel.cu"),
    (1000, 26, "additionMatricesKernel.cu"),
    (1000, 27, "transposedMatrixKernel.cu"),
    (1000, 28, "additionMatricesKernel.cu"),
    (1000, 29, "matrixKernel.cu"),
    (104, 2, "matrixMultiplyTiled.cu"),
    (104, 3, "matrixMultiply.cu"),
    (10426, 0, "gaxpymm.cu"),
    (10426, 1, "gaxpy.cu"),
    (1, 0, "euclidean_kernel.cu"),
    (1, 1, "euclidean_kernel_same.cu"),
    (1, 2, "maximum_kernel.cu"),
    (1, 3, "maximum_kernel_same.cu"),
    (10693, 4, "MatrixMul.cu"),
    (10693, 5, "MatrixMulSh.cu"),
    (10716, 9, "MatrixMul.cu"),
    (1006, 0, "kernel_example.cu"),
    (10242, 0, "kernel.cu"),
    (10220, 1, "hello.cu"),
    (10126, 2, "conv2genericrev.cu"),
    (10270, 2, "conv2genericrev.cu"),
    (15753, 199, "downscale.cu"),
    (15777, 199, "downscale.cu"),
    (1643, 11, "FluffyTail.cu"),
    (16427, 1, "horspool_match.cu"),
    (10790, 97, "NormalizeKernel.cu"),
    (10835, 97, "NormalizeKernel.cu"),
    (10003, 2, "deltaCalcOutput.cu"),
    (10383, 0, "batcherBitonicMergesort64.cu"),
    (11819, 1, "gpu_stencil2D_4pt_hack5_cp_rows.cu"),
    (13126, 0, "leven.cu"),
    (10008, 81, "softmax_gradient_kernel.cu"),
    (11934, 93, "relu_gpu_forward.cu"),
]


if __name__ == "__main__":
    for major_id, minor_id, fname in KERNELS_TO_DOWNLOAD:
        try:
            download_kernel(major_id, minor_id, fname)
            print("  → Success")
        except Exception as e:
            print(f"Failed for ({major_id}, {minor_id}, {fname}): {e}")

