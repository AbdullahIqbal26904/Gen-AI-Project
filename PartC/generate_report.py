#!/usr/bin/env python3
"""Generate Part C report PDF - 2 pages max, images right, text left."""

from fpdf import FPDF
import os

IMG_DIR = os.path.join(os.path.dirname(__file__), "results")
INLINE_DIR = os.path.join(IMG_DIR, "__results___files")
OUT_PATH = os.path.join(os.path.dirname(__file__), "GenAI_Project_PartC_Report.pdf")

LM = 10        # left margin
RM = 10        # right margin
TW = 190       # total usable width (210 - LM - RM)
LEFT_W = 90    # text column width
RIGHT_W = 95   # image column width
GAP = 5        # gap between columns
IMG_X = LM + LEFT_W + GAP  # image column x position


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, "Generative AI Project - Part C: GAN-based Data Augmentation", align="C")
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(20, 60, 120)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(40, 40, 40)
        self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text, w=0):
        self.set_font("Helvetica", "", 8.5)
        self.multi_cell(w if w else 0, 4, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [TW / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 5, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(0, 0, 0)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(230, 240, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for val, w in zip(row, col_widths):
                self.cell(w, 4.5, str(val), border=1, align="C", fill=True)
            self.ln()
            fill = not fill
        self.ln(2)

    def text_image_row(self, text, img_path, img_w=RIGHT_W, img_h=None):
        """Place text on the left and an image on the right, side by side."""
        y_start = self.get_y()
        # Draw text on left column
        self.set_font("Helvetica", "", 8.5)
        self.set_xy(LM, y_start)
        self.multi_cell(LEFT_W, 4, text)
        y_after_text = self.get_y()
        # Draw image on right column
        if os.path.exists(img_path):
            self.image(img_path, x=IMG_X, y=y_start, w=img_w)
            # Estimate image height from aspect ratio
            from PIL import Image
            im = Image.open(img_path)
            aspect = im.size[1] / im.size[0]
            actual_h = img_w * aspect
            y_after_img = y_start + actual_h
        else:
            y_after_img = y_start
        # Move cursor below whichever is taller
        self.set_y(max(y_after_text, y_after_img) + 2)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_left_margin(LM)
    pdf.set_right_margin(RM)
    pdf.set_auto_page_break(auto=True, margin=12)

    # ═══════════════════════ PAGE 1 ═══════════════════════
    pdf.add_page()

    # ── Title row ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 7, "Part C: GAN-based Data Augmentation for CIFAR-10", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 5, "Mahnoor Adeel  |  Hamza Zaman  |  Abdullah Iqbal", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # ── CGAN Training: text left, loss curves right ──
    pdf.section_title("1. CGAN Training & Generated Samples")

    pdf.text_image_row(
        "A Conditional GAN (CGAN) was trained for 100 epochs on the "
        "CIFAR-10 training set (50,000 images). The generator takes a "
        "128-dim noise vector + one-hot class label and upsamples "
        "through Conv2DTranspose layers to produce 32x32x3 images. "
        "The discriminator receives an image concatenated with the "
        "class label and outputs a real/fake probability.\n\n"
        "Both networks use Adam (lr=0.0002, beta_1=0.5) with binary "
        "cross-entropy loss. Label smoothing (real=0.9) is applied "
        "for training stability.\n\n"
        "The loss curves show the generator loss increasing (~0.9 to "
        "~1.5) while the discriminator loss decreases (~0.65 to ~0.53), "
        "indicating active adversarial dynamics without mode collapse.",
        os.path.join(INLINE_DIR, "__results___19_0.png"),
        img_w=92,
    )

    pdf.text_image_row(
        "Generated samples show recognisable structure for most "
        "classes. Vehicles (automobile, truck, ship) and animals with "
        "distinctive shapes (horse, frog) exhibit clear patterns. "
        "Classes with high intra-class variation (bird, cat) appear "
        "blurrier. Compared to VAE outputs from Part B, GAN images "
        "are notably sharper with more realistic textures.",
        os.path.join(INLINE_DIR, "__results___21_0.png"),
        img_w=92,
    )

    # ── Results: accuracy table + val accuracy curve ──
    pdf.section_title("2. Results")

    pdf.add_table(
        ["Model", "Test Accuracy", "Macro F1"],
        [
            ["Baseline", "0.6402", "0.65"],
            ["GAN +10/class", "0.6836", "0.68"],
            ["GAN +50/class", "0.6670", "0.68"],
            ["GAN +100/class", "0.6724", "0.67"],
        ],
        col_widths=[65, 60, 60],
    )

    pdf.text_image_row(
        "All GAN-augmented models outperform the baseline. The best "
        "result is +10 samples/class (68.36%), a +4.34% gain over the "
        "baseline (64.02%). Increasing to 50 or 100 per class yields "
        "slightly lower accuracy (66.70%, 67.24%).\n\n"
        "Validation accuracy curves show all models converging to the "
        "0.65-0.72 range by epoch 30. Augmented models exhibit "
        "marginally smoother convergence.\n\n"
        "Training loss curves are nearly identical across all configs, "
        "with augmented models showing slightly higher loss - a mild "
        "regularisation effect from the synthetic samples.",
        os.path.join(INLINE_DIR, "__results___34_0.png"),
        img_w=92,
    )

    # F1 bar chart and confusion matrix side-by-side context
    pdf.text_image_row(
        "Per-class F1-scores show the most improvement in classes "
        "with distinct visual features: automobile (0.73->0.80), "
        "truck (0.72->0.78), deer (0.50->0.63). Difficult classes "
        "like cat (0.48->0.49) and bird (0.54->0.54) see minimal "
        "change due to high intra-class variability.\n\n"
        "Confusion matrices confirm that GAN augmentation reduces "
        "severe misclassifications - deer correct predictions jump "
        "from 380 to 630 with +10/class. The cat-dog confusion "
        "persists across all configurations due to inherent visual "
        "similarity at 32x32 resolution.",
        os.path.join(INLINE_DIR, "__results___40_0.png"),
        img_w=92,
    )

    # ═══════════════════════ PAGE 2 ═══════════════════════
    pdf.add_page()

    pdf.section_title("3. Discussion: Why and Why Not There Is Improvement")

    pdf.sub_title("3.1 Why GAN Augmentation Improves Accuracy")
    pdf.text_image_row(
        "The GAN-generated images act as a regulariser by introducing "
        "additional training diversity. Unlike the VAE from Part B whose "
        "outputs are blurry pixel-wise averages, adversarial training "
        "produces sharper images with realistic local textures that lie "
        "closer to the true data manifold. This provides more useful "
        "gradient signal during CNN training.\n\n"
        "Even adding just 10 samples per class (100 total, only 0.22% "
        "of the 45,000 training set) yields a meaningful +4.34% accuracy "
        "gain. This suggests GAN samples introduce novel variations "
        "(slightly different poses, backgrounds, colour shifts) that "
        "the CNN benefits from, improving generalisation to unseen test "
        "data. The improvement is most visible in classes where the GAN "
        "captures distinctive features well (vehicles, horse, frog).",
        os.path.join(INLINE_DIR, "__results___42_0.png"),
        img_w=92,
    )

    pdf.sub_title("3.2 Why More Samples Do Not Always Help")
    pdf.text_image_row(
        "Increasing from 10 to 50 or 100 per class does not improve "
        "accuracy further (68.36% -> 66.70% -> 67.24%). Three factors "
        "explain this:\n\n"
        "1. Mode redundancy: The generator has limited mode coverage. "
        "More samples from the same generator increasingly repeat "
        "learned modes, adding redundancy rather than genuine diversity.\n\n"
        "2. Distribution shift: As synthetic-to-real ratio grows, the "
        "CNN's training distribution shifts from the true test "
        "distribution. The CNN may overfit to GAN artefacts "
        "(characteristic blurriness, colour biases) absent in real "
        "test images.\n\n"
        "3. Quality ceiling: For difficult classes (cat, bird), generated "
        "samples introduce more noise than signal. Cat F1-score actually "
        "drops from 0.49 to 0.45 at +100/class, showing that low-quality "
        "synthetic data can hurt specific class performance.",
        os.path.join(INLINE_DIR, "__results___35_0.png"),
        img_w=92,
    )

    pdf.sub_title("3.3 Comparison with VAE Augmentation (Part B)")
    pdf.body_text(
        "GAN augmentation shows a clear advantage over VAE augmentation. The key difference is in the "
        "training objectives: VAEs optimise a pixel-level reconstruction loss that produces blurry outputs "
        "averaging over modes, while GANs use an adversarial loss that encourages sharp, realistic outputs. "
        "As a result, GAN-generated images provide more informative training signal to the CNN. However, "
        "neither approach achieves dramatic improvements because the original CIFAR-10 training set "
        "(5,000 images/class) is already large. Generative data augmentation is most impactful in low-data "
        "regimes - with 50,000 real samples already available, adding a few hundred synthetic ones provides "
        "only marginal gains."
    )

    pdf.section_title("4. Conclusion")
    pdf.body_text(
        "A Conditional GAN was trained on CIFAR-10 and used for class-conditional data augmentation. "
        "The best configuration (GAN +10/class) achieved 68.36% test accuracy, a +4.34% improvement over "
        "the 64.02% baseline. Increasing generated samples to 50 or 100/class did not yield further gains "
        "due to mode redundancy and distribution shift. The GAN approach outperforms Part B's VAE augmentation, "
        "producing sharper synthetic images that better serve the downstream CNN classifier."
    )

    pdf.output(OUT_PATH)
    print(f"Report saved to: {OUT_PATH}")


if __name__ == "__main__":
    build_report()
