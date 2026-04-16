import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 生成测试图像 ==========
def generate_checkerboard(width, height, cell_size):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            cx = x // cell_size
            cy = y // cell_size
            if (cx + cy) % 2 == 0:
                img[y, x] = 255
    return img

def generate_radial_chirp(width, height, k):
    img = np.zeros((height, width), dtype=np.uint8)
    cx, cy = width/2, height/2
    max_r2 = cx*cx + cy*cy
    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            r2 = dx*dx + dy*dy
            phase = 2 * np.pi * k * r2 / max_r2
            val = 127 + 127 * np.sin(phase)
            img[y, x] = int(val)
    return img

# ========== 2. 显示频谱（中心化+对数）使用 matplotlib ==========
def show_spectrum(img, title, filename):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    magnitude = np.log(1 + magnitude)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)
    plt.figure(figsize=(6,6))
    plt.imshow(magnitude, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"保存频谱图: {filename}")

# ========== 3. 保存普通图像 ==========
def save_image(img, filename, title):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"保存图像: {filename}")

# ========== 4. 下采样函数 ==========
def downsample_direct(img, scale):
    h, w = img.shape
    new_h, new_w = int(h*scale), int(w*scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def downsample_with_gaussian(img, scale, sigma):
    blurred = cv2.GaussianBlur(img, (0,0), sigma)
    h, w = img.shape
    new_h, new_w = int(h*scale), int(w*scale)
    return cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# ========== 5. 自适应下采样 ==========
def adaptive_downsample(img, scale, block_size=32):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-5)
    
    h, w = img.shape
    sigma_map = np.zeros((h, w), dtype=np.float32)
    sigma_min, sigma_max = 0.5, 2.5
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y+block_size, h)
            x_end = min(x+block_size, w)
            block_grad = grad_mag[y:y_end, x:x_end]
            mean_grad = np.mean(block_grad)
            sigma = sigma_min + mean_grad * (sigma_max - sigma_min)
            sigma_map[y:y_end, x:x_end] = sigma
    
    result = np.zeros_like(img, dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y+block_size, h)
            x_end = min(x+block_size, w)
            block = img[y:y_end, x:x_end]
            sigma = np.mean(sigma_map[y:y_end, x:x_end])
            blurred = cv2.GaussianBlur(block, (0,0), sigma)
            result[y:y_end, x:x_end] = blurred
    new_h, new_w = int(h*scale), int(w*scale)
    return cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# ========== 主程序 ==========
def main():
    width, height = 512, 512
    M = 4.0
    scale = 1.0 / M

    # 生成测试图
    checker = generate_checkerboard(width, height, 16)
    chirp = generate_radial_chirp(width, height, 4.0)
    save_image(checker, "checker.png", "棋盘格原图")
    save_image(chirp, "chirp.png", "Chirp原图")

    # 第一部分：直接下采样 vs 高斯滤波后下采样
    print("第一部分：观察混叠与抗混叠...")
    checker_direct = downsample_direct(checker, scale)
    checker_gaussian = downsample_with_gaussian(checker, scale, sigma=1.8)
    save_image(checker_direct, "checker_direct.png", "棋盘格-直接下采样")
    save_image(checker_gaussian, "checker_gaussian.png", "棋盘格-高斯滤波后下采样")

    show_spectrum(checker, "棋盘格原图频谱", "checker_spectrum_original.png")
    show_spectrum(checker_direct, "棋盘格直接下采样频谱", "checker_spectrum_direct.png")
    show_spectrum(checker_gaussian, "棋盘格抗混叠后频谱", "checker_spectrum_gaussian.png")

    chirp_direct = downsample_direct(chirp, scale)
    chirp_gaussian = downsample_with_gaussian(chirp, scale, sigma=1.8)
    save_image(chirp_direct, "chirp_direct.png", "Chirp-直接下采样")
    save_image(chirp_gaussian, "chirp_gaussian.png", "Chirp-高斯滤波后下采样")
    show_spectrum(chirp, "Chirp原图频谱", "chirp_spectrum_original.png")
    show_spectrum(chirp_direct, "Chirp直接下采样频谱", "chirp_spectrum_direct.png")
    show_spectrum(chirp_gaussian, "Chirp抗混叠后频谱", "chirp_spectrum_gaussian.png")

    # 第二部分：不同 sigma 测试
    print("第二部分：固定 M=4，测试 sigma=0.5,1.0,2.0,4.0")
    sigmas = [0.5, 1.0, 2.0, 4.0]
    for sigma in sigmas:
        down = downsample_with_gaussian(checker, scale, sigma)
        save_image(down, f"checker_sigma_{sigma}.png", f"sigma={sigma}")
    print(f"理论最优 sigma = 0.45 * M = {0.45 * M}")

    # 第三部分：自适应下采样
    print("第三部分：自适应下采样 vs 统一 sigma 下采样")
    adaptive = adaptive_downsample(chirp, scale, block_size=32)
    uniform = downsample_with_gaussian(chirp, scale, sigma=1.8)
    save_image(adaptive, "adaptive_down.png", "自适应下采样结果")
    save_image(uniform, "uniform_down.png", "统一 sigma 下采样结果")

    # 误差图（与直接下采样对比）
    diff_uniform = cv2.absdiff(chirp_direct, uniform) * 4
    diff_adaptive = cv2.absdiff(chirp_direct, adaptive) * 4
    save_image(diff_uniform, "error_uniform.png", "统一sigma误差图")
    save_image(diff_adaptive, "error_adaptive.png", "自适应sigma误差图")

    print("全部完成！所有图片已保存到当前目录。")

if __name__ == "__main__":
    main()
