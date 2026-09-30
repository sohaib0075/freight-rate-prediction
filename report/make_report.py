"""Build the submission report (PDF).

    python report/make_report.py     # -> report/freight_rate_report.pdf
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from PIL import Image as PILImage
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
INK = colors.HexColor("#064A56")
ACCENT = colors.HexColor("#C2703D")
MUTED = colors.HexColor("#455A60")
RULE = colors.HexColor("#D9E2E4")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=19, leading=23, textColor=INK, alignment=TA_LEFT,
                            spaceAfter=2),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, leading=13,
                          textColor=MUTED, spaceAfter=10),
    "h": ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold",
                        fontSize=12.5, leading=15, textColor=INK,
                        spaceBefore=13, spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                         fontSize=10, leading=13, textColor=MUTED,
                         spaceBefore=8, spaceAfter=3),
    "b": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.4, leading=13.4,
                        spaceAfter=6),
    "li": ParagraphStyle("li", parent=ss["BodyText"], fontSize=9.4, leading=13.4,
                         leftIndent=11, bulletIndent=2, spaceAfter=3.5),
    "cap": ParagraphStyle("cap", parent=ss["Normal"], fontSize=8.2, leading=10.5,
                          textColor=MUTED, spaceBefore=2, spaceAfter=9),
}


def para(text, style="b"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(f"<bullet>&bull;</bullet>{t}", S["li"]) for t in items]


def figure(name, caption, width=165 * mm):
    path = (FIG / name).resolve()
    if not path.is_file():
        path = (ROOT / name).resolve()
    with PILImage.open(path) as handle:
        pixel_width, pixel_height = handle.size
    img = Image(str(path), width=width, height=width * pixel_height / pixel_width)
    img.hAlign = "LEFT"
    return KeepTogether([img, para(caption, "cap")])


def table(rows, widths, highlight_row=None, align_right_from=1):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.4),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
    ]
    if highlight_row is not None:
        style += [("BACKGROUND", (0, highlight_row), (-1, highlight_row),
                   colors.HexColor("#EEF4F5")),
                  ("FONT", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold", 8.4)]
    t.setStyle(TableStyle(style))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 11.5 * mm, "Freight Rate Prediction - ML Engineer Assessment")
    canvas.drawRightString(190 * mm, 11.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build(story, out):
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=17 * mm, bottomMargin=20 * mm,
                          title="Freight Rate Prediction - ML Engineer Assessment")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=header_footer)])
    doc.build(story)


def main() -> None:
    metrics = json.loads((HERE / "validation_metrics.json").read_text())
    screened = json.loads((HERE / "screened_metrics.json").read_text())
    summary = json.loads((HERE / "model_summary.json").read_text())

    fwd = metrics["forward_sep_oct"]
    roll = metrics["rolling"]
    scr = screened["labels_passing_the_screen"]
    allr = screened["all_rows"]
    day = summary["day_level_fit"]
    dec = summary["december"]

    story = []
    story += [
        para("Freight Rate Prediction", "title"),
        para("Machine Learning Engineer assessment &mdash; modelling approach, validation design "
             "and results. 48,000 labelled loads (Jan&ndash;Oct 2025) are used to predict 12,000 "
             "loads in Nov&ndash;Dec 2025, plus a fixed-load December rate curve.", "sub"),
    ]

    # ---------------------------------------------------------------- summary
    story += [para("1. Summary", "h")]
    story += [para(
        f"The model predicts the Sep&ndash;Oct forward holdout with a <b>median absolute error of "
        f"{fwd['MedAPE_%']:.2f}%</b> (${fwd['MedAE']:.2f} on a typical load). On the "
        f"{100 - screened['screen_rejected_%']:.0f}% of holdout rows whose label passes a corruption "
        f"screen it reaches <b>MAE ${scr['MAE']:.2f}, RMSE ${scr['RMSE']:.2f}, MAPE "
        f"{scr['MAPE_%']:.2f}%, R&sup2; {scr['R2']:.4f}</b>. Across all rows, including the "
        f"~1.3% whose labels are corrupted in the source data and are unpredictable by "
        f"construction, MAE is ${allr['MAE']:.2f} and R&sup2; is {allr['R2']:.3f}.")]
    story += bullets([
        "<b>Target.</b> Log rate-per-mile. The rate is close to multiplicative in distance, "
        "equipment and the market, and the label corruption is multiplicative too.",
        "<b>Structure.</b> Two parts. A gradient-boosted model prices the load within its day; "
        "a four-coefficient regression places the day in the market. Only the second part has to "
        "extrapolate, and it is deliberately small enough to do so safely.",
        "<b>Split.</b> Forward in time, always. A random K-fold would put loads from the same day "
        "on both sides of the split and report a score the model cannot reproduce on a real "
        "future month.",
    ])

    story += [para("2. The data", "h")]
    story += [para(
        "48,000 training loads spanning 2025-01-01 to 2025-10-31 and 12,000 validation loads "
        "spanning 2025-11-01 to 2025-12-31 &mdash; the validation set is strictly in the future, "
        "and it is two full months, not a random sample. Each load carries origin and destination "
        "(64 cities, each with one fixed lat/lon), distance, equipment "
        "(Dry Van 57%, Reefer 25%, Flatbed 18%), weight, date, and two market series: "
        "<b>market_index</b> and <b>quote_signal</b>.")]
    story += bullets([
        "<b>market_index is a daily market series.</b> Within-day sd is 0.025 against an overall "
        "sd of 0.166 &mdash; each day has a level, and each row a small deviation from it. Both "
        "matter: the day level carries a coefficient of "
        f"{day['coef']['log_market_index_daily']:.3f} on log rate-per-mile, and a row one sd above "
        "its own day still runs ~0.3% dearer.",
        "<b>quote_signal is noisier but not noise.</b> Its daily mean moves the rate level with a "
        f"coefficient of {day['coef']['quote_signal_daily']:.3f} (t = "
        f"{day['t_stat']['quote_signal_daily']:.0f}), and the row-level deviation adds a further "
        "monotone effect.",
        "<b>Distance is noisy, and the rate was generated from the noisy value.</b> The same lane "
        "shows ~2% sd in reported distance, but replacing distance with a de-noised per-lane "
        "median makes the fit <i>worse</i> (residual sd 0.0493 against 0.0456). So distance is "
        "used exactly as recorded.",
        "<b>12% of validation loads touch cities that never appear in training</b> (San Diego, "
        "Laredo, Jackson, Charlotte, Chicago, Knoxville, Norfolk, Allentown). Every geographic "
        "signal is therefore expressed in coordinates as well as in city names, so the model "
        "interpolates to them; the nearest known market is 1&ndash;230 miles away. Predicted "
        "rate-per-mile on those rows (mean 2.222) matches the rest of the set (2.211).",
    ])

    story += [para("3. Data-quality issues, and what was done about them", "h")]
    story += [table([
        ["Issue", "Extent", "Treatment"],
        ["weight sign-flipped", "292 train / 145 valid",
         "absolute value — |w| matches the positive distribution exactly"],
        ["weight missing", "300 train / 165 valid", "median impute + missing flag"],
        ["market_index missing", "374 train / 249 valid",
         "same-day mean of the other loads, then global median"],
        ["posted_rate corrupted", "~1.3% of labels",
         "detected out-of-fold, dropped before the final fit"],
        ["August labels noisier", "whole month",
         "month down-weighted to 0.35 by inverse residual variance"],
    ], [38 * mm, 33 * mm, 94 * mm], align_right_from=3)]
    story += [para("", "cap")]
    story += [para(
        "The last two are the ones that actually cost accuracy. <b>Label corruption:</b> about 1.3% "
        "of posted_rate values are multiplied by a random factor &mdash; roughly 1.4&times; to "
        "6.6&times; up, or 0.15&times; to 0.7&times; down. That is one to two orders of magnitude "
        "larger than the genuine residual spread (sd &asymp; 0.015 in logs), so the two "
        "populations separate cleanly. They are flagged from the out-of-fold residuals of a "
        "date-blind model and dropped: "
        f"{summary['rows_dropped_as_corrupt_pct']:.2f}% of training rows, leaving "
        f"{summary['n_train_rows_used']:,}. Dropping is right rather than clipping, because the "
        "corruption carries no information about the true rate.")]
    story += [figure("data_quality.png",
                     "Left: negating the negative weights reproduces the positive "
                     "distribution (n = 292, hence the jitter), which is what identifies them as "
                     "sign flips rather than credits or returns. Right: log rate-per-mile has a "
                     "tight body and two far-flung shoulders; the shaded band is what is kept.")]
    story += [para(
        "<b>August:</b> residual spread in August is 0.026 against ~0.013 in every other month "
        "&mdash; uniformly across days, equipment types and haul lengths, so it is extra noise on "
        "the labels rather than anything a feature explains. Nothing repairs that, but it is "
        "heteroskedasticity, so each month is weighted by the inverse of its residual variance. "
        "August lands at 0.35, and held-out cross-sectional error improves 2&ndash;5%. It also "
        "means the August holdout below is a poor guide to model quality, and it is read as such.")]
    story += [figure("month_noise.png",
                     "Robust residual sd by month, from an out-of-fold cross-sectional fit. "
                     "August stands alone.", 118 * mm)]

    # ------------------------------------------------------------------ model
    story += [para("4. Model", "h")]
    story += [para(
        "Variance in log rate-per-mile splits into a within-day part (sd &asymp; 0.018 &mdash; "
        "which load is it?) and a day-level part (sd &asymp; 0.030 &mdash; what was the market "
        "doing?). The two need different machinery, so they get it.")]
    story += [para("4.1 Cross-section &mdash; gradient boosting", "h3")]
    story += [para(
        "LightGBM on log rate-per-mile, 63 leaves, learning rate 0.03, feature fraction 0.7, "
        "1,500 rounds, averaged over four seeds. Features: distance and log distance, weight and "
        "weight-per-mile, equipment, both cities as categoricals, both coordinate pairs plus "
        "midpoint, deltas, great-circle distance, detour ratio and bearing, lane frequency, "
        "day-of-week and day-of-month, and the row's deviation from its own day's market index "
        "and quote signal. Everything is tuned against <i>within-day</i> error, which is free of "
        "the day-level forecasting problem and robust to the corrupted labels.")]
    story += [para("4.2 Day level &mdash; four coefficients, fitted by weighted least squares", "h3")]
    story += [para(
        "This is the only part that extrapolates, and the design is driven by that. Trees cannot "
        "extrapolate a trend at all; worse, when handed day aggregates such as the daily mean "
        "quote signal they memorise <i>which month</i> a day belongs to and then fan out on a new "
        "month whose aggregates sit in a narrow band. That was measured, not assumed: giving the "
        "trees the daily quote mean tripled the median error on an August holdout &mdash; and "
        "November and December have exactly the narrow-band shape that triggers it.")]
    story += [para(
        "So the day level is a linear regression on the daily mean of the cross-sectional model's "
        "out-of-fold residual, weighted by each day's load count and by its month's reliability:")]
    story += [table([
        ["Day-level term", "Coefficient", "robust t"],
        ["log(market_index), daily", f"{day['coef']['log_market_index_daily']:+.4f}",
         f"{day['t_stat']['log_market_index_daily']:.1f}"],
        ["quote_signal, daily mean", f"{day['coef']['quote_signal_daily']:+.4f}",
         f"{day['t_stat']['quote_signal_daily']:.1f}"],
        ["calendar time (per day)", f"{day['coef']['t']:+.5f}", f"{day['t_stat']['t']:.1f}"],
    ], [64 * mm, 34 * mm, 26 * mm])]
    story += [para("", "cap")]
    story += [para(
        f"The calendar-time coefficient is the secular drift: <b>"
        f"{day['drift_per_100_days']:+.4f} log-units per 100 days</b>, about "
        f"{100 * (math.exp(day['coef']['t'] * 365.25) - 1):+.1f}% a year, on top of everything the "
        "market index explains. It is what makes the Nov&ndash;Dec level right. Zero that one "
        "coefficient and the same model, refitted, predicts the average of its training window "
        "instead: on the Sep&ndash;Oct holdout the median bias goes from &minus;0.51% to "
        "&minus;2.40% and MedAPE more than doubles, from 1.06% to 2.44%. A window two months past "
        "the training data, like the real one, is further out still. "
        "Out-of-fold residuals are used to fit it because in-sample GBM residuals are shrunk and "
        "would bias the drift toward zero.")]
    story += [para(
        "The specification was chosen by backtesting the daily level series on its own "
        "(<font face='Courier'>experiments/day_level_backtest.py</font>): roll an origin forward "
        "every week, fit on everything before it, forecast the next 30 and 61 days, score. Three "
        "things come out of it, all quoted below at 61 days &mdash; the real horizon &mdash; and "
        "averaged over the origins with seven months or more of history, which are the ones that "
        "resemble the real task.")]
    story += bullets([
        "<b>Calendar time is doing the work.</b> Level RMSE 0.0102 with it, 0.027&ndash;0.029 "
        "without &mdash; a factor of three, and nothing else recovers it.",
        "<b>Day-of-week and day-of-month belong in the trees, not here.</b> Adding them to the "
        "day model wins on the average over all twenty origins (0.0124 against 0.0197), but that "
        "average is carried by the June&ndash;July origins where every specification is poor. At "
        "the realistic origins they are the worst of the set (0.0130 against 0.0102), and the gap "
        "opens up with horizon &mdash; at 30 days they are level with it. Their coefficients are "
        "not stable enough to carry two months.",
        "<b>The quote term is a genuine coin flip.</b> Dropping it scores 0.0098 against 0.0102 "
        "at the realistic origins, and 0.0118 against 0.0116 on the median over all of them. It "
        "is kept because it is strongly significant in-sample and costs nothing either way: "
        "November and December sit almost exactly at the training mean of the quote series, so it "
        "moves the final level by about 0.02%.",
    ])
    story += [para(
        "The error also does not grow with horizon &mdash; 0.0105 at 30 days against 0.0102 at 61 "
        "&mdash; which is the evidence that the drift is a genuine linear trend rather than a "
        "local slope that happens to fit the end of the training window.")]
    story += [figure("day_level.png",
                     "The grey line is the observed daily level, the dark line the fitted day "
                     "model, the orange line the same four coefficients applied to Nov-Dec using "
                     "the market and quote series supplied for those dates. The dotted line is "
                     "the drift alone.")]

    # ------------------------------------------------------- split/validation
    story += [para("5. Validation and data split", "h")]
    story += [para(
        "<b>Every split is forward in time.</b> Once distance, equipment and geography are "
        "accounted for, most of what remains is a day-level market effect (sd 0.030, against "
        "0.018 within a day), so a random K-fold would put loads from the same day on both sides "
        "of the split and hand the model much of the answer; it would also hide the drift, "
        "since each fold would contain the full date range. Neither failure would show up in the "
        "K-fold score, and both would show up in production.")]
    story += bullets([
        "<b>Primary &mdash; forward holdout.</b> Train Jan&ndash;Aug, score Sep&ndash;Oct. Two "
        "unseen months, the same horizon as the real Nov&ndash;Dec task.",
        "<b>Stability &mdash; rolling origin.</b> Four successive 1-month holdouts (Jul, Aug, Sep, "
        "Oct), each trained only on what precedes it.",
        "<b>Inside the training fold</b> the model fits its own out-of-fold residuals twice &mdash; "
        "once to flag corrupted labels, once to build the day-level series. Both use random folds, "
        "which is safe precisely because neither model carries a day-level feature, so rows from "
        "the same day cannot leak a level to each other.",
        "<b>Model selection was never done on the final validation set</b>, which has no labels. "
        "Cross-sectional choices were scored on within-day error; the day-level specification was "
        "scored on its own rolling forecast backtest.",
    ])
    story += [para(
        "One note on what is shared across the split. Daily market aggregates, the city "
        "coordinate table, lane counts and the category lists are computed over the training and "
        "validation <i>feature</i> columns together. No label is involved, and the validation "
        "features are given as part of the task; this is what makes the December chart possible "
        "at all, since those 31 rows arrive with no market data of their own.")]

    story += [para("6. Results", "h")]
    story += [para(
        "Sep&ndash;Oct forward holdout, 9,523 loads. The two rows differ only in whether the "
        "~1.3% of rows with corrupted labels are included. The screen is built inside the holdout "
        "from its own loads and labels, never from the model being scored &mdash; and MedAPE "
        f"barely moves ({allr['MedAPE_%']:.3f}% to {scr['MedAPE_%']:.3f}%), which is how you can "
        "tell it is removing bad labels rather than flattering the model. The mean-based metrics "
        "move a great deal, because a handful of 6&times; labels dominate a squared error.")]
    story += [table([
        ["Sep-Oct holdout", "MAE", "RMSE", "MAPE", "MedAPE", "R²"],
        ["all 9,523 rows", f"${allr['MAE']:.2f}", f"${allr['RMSE']:.2f}",
         f"{allr['MAPE_%']:.2f}%", f"{allr['MedAPE_%']:.2f}%", f"{allr['R2']:.3f}"],
        [f"labels passing the screen ({scr['n']:,})", f"${scr['MAE']:.2f}", f"${scr['RMSE']:.2f}",
         f"{scr['MAPE_%']:.2f}%", f"{scr['MedAPE_%']:.2f}%", f"{scr['R2']:.4f}"],
    ], [58 * mm, 22 * mm, 24 * mm, 22 * mm, 23 * mm, 20 * mm], highlight_row=2)]
    story += [para("", "cap")]
    story += [para(
        "Rolling-origin backtest, median absolute percentage error on every row, and the same "
        "error split into its day-level and within-day halves:")]
    rows = [["Holdout month", "MedAPE", "MedAE", "median bias", "day-level RMSE", "within-day MedAE"]]
    for key, r in roll.items():
        rows.append([datetime.strptime(key, "%Y-%m").strftime("%B"),
                     f"{r['MedAPE_%']:.2f}%", f"${r['MedAE']:.2f}",
                     f"{r['median_bias_%']:+.2f}%", f"{r['day_level_rmse']:.4f}",
                     f"{r['within_day_MedAE']:.4f}"])
    story += [table(rows, [30 * mm, 21 * mm, 21 * mm, 24 * mm, 29 * mm, 33 * mm])]
    story += [para("", "cap")]
    story += [para(
        "September and October &mdash; the two origins with the most history, and so the closest "
        f"analogues to the real task &mdash; land at {roll['2025-09']['MedAPE_%']:.2f}% and "
        f"{roll['2025-10']['MedAPE_%']:.2f}% with biases under 0.6%. July's day level is harder: "
        "fitted on only six months, the drift estimate is distorted by a mid-year bump that later "
        "reverts. August is the noisy month described in section 3; its within-day error (0.031) "
        "is three times every other month's and is a property of the labels, not the model. "
        "Both effects shrink as history grows, and the final model has ten months of it.")]
    story += [figure("holdouts.png",
                     "Left: error by holdout month. Right: the same error split in two. The "
                     "within-day half is flat and small except in August; the day-level half is "
                     "what improves as the training window lengthens.")]

    # -------------------------------------------------------------- december
    story += [para("7. The fixed December chart", "h")]
    story += [para(
        "Thirty-one identical loads &mdash; Lexington to Fort Wayne, 360 miles, Dry Van, "
        "32,000 lb &mdash; one per December day. Only the date changes, so the curve <i>is</i> "
        "the day-level term, and it is a direct read-out of the model's view of the December "
        "market. Coordinates come from the city table; market_index and quote_signal come from "
        "the daily means of the validation loads on those dates, which is where the task supplies "
        "December market conditions. The lane itself is well attested: 44 loads in the data, "
        "reported distances averaging 365 miles, so 360 needs no extrapolation.")]
    story += [figure("../../scorer_results/candidate_december.png",
                     f"Produced by the provided score.py. Predicted rates run "
                     f"${min(dec.values()):.2f} to ${max(dec.values()):.2f}.")]
    story += [para(
        "The shape is a weekly cycle on a slight upward drift. Mid-week peaks and weekend troughs "
        "come from market_index, which carries a strong weekly pattern; the month-over-month rise "
        "is the drift term. Both are properties the model learned from ten months of data, not "
        "assumptions imposed on December.")]

    story += [para("8. Limits, and what I would do next", "h")]
    story += bullets([
        "<b>The drift is one year of one series.</b> Ten months cannot separate a secular trend "
        "from slow annual seasonality; the linear term is the defensible reading and the backtest "
        "supports it out to 61 days, but a second year of history would settle it. The exposure "
        "is bounded: over the Nov&ndash;Dec window the drift term contributes about 1.3%.",
        "<b>The day level still has 0.008&ndash;0.010 of unforecastable wobble.</b> That is now "
        "comparable to the entire within-day error, so it is where the next real gain is &mdash; "
        "most likely from market features the dataset does not contain (fuel, capacity, holidays) "
        "rather than from more modelling of what is here.",
        "<b>Predictions are conditional medians.</b> With residual sd ~0.015 the median-to-mean "
        "gap is ~0.01%, so it is not worth a smearing correction. If the scorer used a squared "
        "error over the raw, still-corrupted labels, a ~1.4% uplift would score marginally "
        "better; that is fitting the corruption rather than the freight market, so it was left "
        "out deliberately.",
        "<b>Unseen cities are interpolated, not learned.</b> Coordinates carry them, and their "
        "predicted rate-per-mile matches the rest of the set, but a genuinely new region far from "
        "any training city would be an extrapolation the validation here cannot vouch for.",
    ])

    out = HERE / "freight_rate_report.pdf"
    build(story, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
