import matplotlib.pyplot as plt

# --- Data Extracted from your Table 2 ---
models = [
    'U-net',
    'U-net quantized',
    'DeeplabV3+ (ResNet101)',
    'DeeplabV3+ (ResNet34)',
    'DeeplabV3+ (Mobilenet_v2)',
    'DeeplabV3+ (Efficientnet_b0)'
]

dice_scores = [0.574, 0.575, 0.605, 0.595, 0.608, 0.614]
cpu_speed = [4.26, 24.15, 7.40, 15.09, 16.43, 13.23]
# GPU speed - Note: Quantized U-Net has no value, EfficientNet is very low
# We'll handle the missing value by filtering or using placeholder if needed
gpu_speed = [161.98, None, 165.92, 269.32, 420.23, 18.06]
gmacs = [161.05, 0.35, 63.50, 37.18, 12.46, 7.62] # Lower is better

# --- Plot 1: Accuracy vs. GPU Inference Speed ---

# Filter out models without GPU speed data
models_gpu = [m for m, g in zip(models, gpu_speed) if g is not None]
dice_gpu = [d for d, g in zip(dice_scores, gpu_speed) if g is not None]
speed_gpu = [g for g in gpu_speed if g is not None]

fig1, ax1 = plt.subplots(figsize=(10, 6)) # Adjust figure size if needed

# Scatter plot
ax1.scatter(speed_gpu, dice_gpu, s=50) # s is marker size

# Add labels to each point
for i, model_name in enumerate(models_gpu):
    # Slightly offset labels for clarity
    ax1.text(speed_gpu[i] * 1.01, dice_gpu[i], model_name, fontsize=9)

# Add axis labels and title
ax1.set_xlabel('GPU Inference Speed (Images/sec) -> Faster')
ax1.set_ylabel('Accuracy (DICE Score - Local Test Set) -> Better')
ax1.set_title('Model Accuracy vs. GPU Inference Speed')

# Add grid for better readability
ax1.grid(True, linestyle='--', alpha=0.6)

# Optional: Adjust axis limits if points are too close to edge
# ax1.set_xlim(left=0)
# ax1.set_ylim(bottom=0.55) # Adjust as needed

plt.tight_layout() # Adjust layout to prevent labels overlapping axes
plt.show()


# --- Plot 2: Accuracy vs. CPU Inference Speed ---

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Scatter plot - All models have CPU data
ax2.scatter(cpu_speed, dice_scores, s=50)

# Add labels to each point
for i, model_name in enumerate(models):
    ax2.text(cpu_speed[i] * 1.01, dice_scores[i], model_name, fontsize=9)

# Add axis labels and title
ax2.set_xlabel('CPU Inference Speed (Images/sec) -> Faster')
ax2.set_ylabel('Accuracy (DICE Score - Local Test Set) -> Better')
ax2.set_title('Model Accuracy vs. CPU Inference Speed')

# Add grid
ax2.grid(True, linestyle='--', alpha=0.6)

# Optional: Adjust axis limits
# ax2.set_xlim(left=0)
# ax2.set_ylim(bottom=0.55)

plt.tight_layout()
plt.show()

# --- (Optional) Plot 3: Accuracy vs. Computational Complexity (GMACs) ---
# Note: Lower GMACs is better

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Scatter plot
ax3.scatter(gmacs, dice_scores, s=50)

# Add labels
for i, model_name in enumerate(models):
     # Special handling for potentially very low GMACs of quantized U-Net if scale is linear
     offset_x = gmacs[i] * 0.05 if gmacs[i] > 1 else 0.5 # Adjust offset logic as needed
     ax3.text(gmacs[i] + offset_x, dice_scores[i], model_name, fontsize=9)


# Add axis labels and title
ax3.set_xlabel('Computational Complexity (GMACs) -> More Efficient (Lower is Better)')
ax3.set_ylabel('Accuracy (DICE Score - Local Test Set) -> Better')
ax3.set_title('Model Accuracy vs. Computational Complexity')

# Consider log scale for X-axis if GMACs vary greatly (like 161 vs 0.35)
# ax3.set_xscale('log')
# ax3.set_xlabel('Computational Complexity (GMACs - Log Scale) -> More Efficient')


# Add grid
ax3.grid(True, linestyle='--', alpha=0.6)

# Optional: Adjust limits
# ax3.set_ylim(bottom=0.55)

plt.tight_layout()
plt.show()
