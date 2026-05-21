from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path('reports')
PRIOR_DAILY = REPORTS_DIR / 'backtest_ens_champion_segmented6_prior_daily.csv'
BASE_DAILY = REPORTS_DIR / 'backtest_ens_champion_segmented6_daily.csv'
BID_DAILY = REPORTS_DIR / 'bid_space_daily_dispatch.csv'
OUT_MD = REPORTS_DIR / 'current_champion_daily_error.md'
OUT_CSV = REPORTS_DIR / 'current_champion_daily_error.csv'


def markdown_table(df: pd.DataFrame, *, floatfmt: str = '.4f') -> str:
    if df.empty:
        return '(empty)'
    cols = list(df.columns)

    def fmt(v: object) -> str:
        if isinstance(v, pd.Timestamp):
            return v.date().isoformat()
        if isinstance(v, float):
            if np.isnan(v):
                return 'nan'
            return format(v, floatfmt)
        return str(v)

    lines = ['| ' + ' | '.join(cols) + ' |']
    lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def load_prior() -> pd.DataFrame:
    df = pd.read_csv(PRIOR_DAILY, parse_dates=['date'])
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.dayofweek
    df['abs_charge_gap'] = df['charge_start_gap'].abs()
    df['abs_discharge_gap'] = df['discharge_start_gap'].abs()
    df['both_gap_sum'] = df['abs_charge_gap'] + df['abs_discharge_gap']
    df['loss_day'] = df['profit'] < 0
    df['within_2_slots'] = (df['abs_charge_gap'] <= 2) & (df['abs_discharge_gap'] <= 2)
    df['within_4_slots'] = (df['abs_charge_gap'] <= 4) & (df['abs_discharge_gap'] <= 4)
    return df


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('fold', as_index=False)
        .agg(
            days=('date', 'count'),
            mean_profit=('profit', 'mean'),
            min_profit=('profit', 'min'),
            mean_oracle_profit=('oracle_profit', 'mean'),
            mean_regret=('regret', 'mean'),
            p90_regret=('regret', lambda s: s.quantile(0.9)),
            loss_days=('loss_day', 'sum'),
            mean_abs_charge_gap=('abs_charge_gap', 'mean'),
            mean_abs_discharge_gap=('abs_discharge_gap', 'mean'),
            exact_charge_hit=('charge_start_gap', lambda s: (s == 0).mean()),
            exact_discharge_hit=('discharge_start_gap', lambda s: (s == 0).mean()),
            within_2_rate=('within_2_slots', 'mean'),
            within_4_rate=('within_4_slots', 'mean'),
        )
        .sort_values('fold')
    )


def gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in [('charge', 'charge_start_gap'), ('discharge', 'discharge_start_gap')]:
        s = df[col]
        rows.append(
            {
                'window': name,
                'mean_gap': s.mean(),
                'median_gap': s.median(),
                'mean_abs_gap': s.abs().mean(),
                'exact_hit_rate': (s == 0).mean(),
                'early_rate': (s < 0).mean(),
                'late_rate': (s > 0).mean(),
                'within_2_rate': (s.abs() <= 2).mean(),
                'within_4_rate': (s.abs() <= 4).mean(),
                'max_abs_gap': s.abs().max(),
            }
        )
    return pd.DataFrame(rows)


def prior_delta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not BASE_DAILY.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = pd.read_csv(BASE_DAILY, parse_dates=['date'])
    keep = [
        'date',
        'profit',
        'charge_start',
        'discharge_start',
        'regret',
        'charge_start_gap',
        'discharge_start_gap',
    ]
    merged = df[keep + ['fold']].merge(base[keep], on='date', suffixes=('_prior', '_base'))
    merged['profit_delta_prior_minus_base'] = merged['profit_prior'] - merged['profit_base']
    merged['regret_delta_prior_minus_base'] = merged['regret_prior'] - merged['regret_base']
    merged['window_changed'] = (
        (merged['charge_start_prior'] != merged['charge_start_base'])
        | (merged['discharge_start_prior'] != merged['discharge_start_base'])
    )
    summary = pd.DataFrame(
        [
            {
                'days': len(merged),
                'changed_days': int(merged['window_changed'].sum()),
                'improved_days': int((merged['profit_delta_prior_minus_base'] > 0).sum()),
                'worsened_days': int((merged['profit_delta_prior_minus_base'] < 0).sum()),
                'same_days': int((merged['profit_delta_prior_minus_base'] == 0).sum()),
                'mean_delta': merged['profit_delta_prior_minus_base'].mean(),
                'median_delta': merged['profit_delta_prior_minus_base'].median(),
                'total_delta': merged['profit_delta_prior_minus_base'].sum(),
            }
        ]
    )
    top_gain = merged.sort_values('profit_delta_prior_minus_base', ascending=False).head(12)
    top_loss = merged.sort_values('profit_delta_prior_minus_base', ascending=True).head(12)
    return summary, top_gain, top_loss


def bid_space_join(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BID_DAILY.exists():
        return pd.DataFrame(), pd.DataFrame()
    bid = pd.read_csv(BID_DAILY, parse_dates=['date'])
    cols = [
        'date',
        'window_spearman_fct_price',
        'window_spearman_act_price',
        'bid_window_error_mae_day',
        'profit_by_bid_fct_pair',
        'profit_by_bid_act_pair',
        'regret_bid_fct_pair',
        'regret_bid_act_pair',
    ]
    merged = df.merge(bid[cols], on='date', how='left')
    corr_rows = []
    for col in cols[1:]:
        corr_rows.append(
            {
                'feature': col,
                'spearman_with_champion_regret': merged[col].corr(
                    merged['regret'], method='spearman'
                ),
                'pearson_with_champion_regret': merged[col].corr(merged['regret']),
                'mean_on_loss_days': merged.loc[merged['loss_day'], col].mean(),
                'mean_on_non_loss_days': merged.loc[~merged['loss_day'], col].mean(),
            }
        )
    return merged, pd.DataFrame(corr_rows).sort_values(
        'spearman_with_champion_regret', ascending=False
    )


def write_report() -> None:
    df = load_prior()
    fsum = fold_summary(df)
    gsum = gap_summary(df)
    loss_days = df[df['loss_day']].sort_values('profit')
    high_regret = df.sort_values('regret', ascending=False).head(20)
    low_ratio = df.sort_values('profit_ratio_day', ascending=True).head(20)
    prior_sum, top_gain, top_loss = prior_delta(df)
    bid_join, bid_corr = bid_space_join(df)
    if not bid_join.empty:
        bid_join.to_csv(OUT_CSV, index=False)

    keep_day = [
        'date',
        'fold',
        'profit',
        'oracle_profit',
        'regret',
        'profit_ratio_day',
        'charge_start',
        'discharge_start',
        'oracle_charge_start',
        'oracle_discharge_start',
        'charge_start_gap',
        'discharge_start_gap',
        'predicted_spread',
        'top1_top5_mean_gap',
    ]
    delta_keep = [
        'date',
        'fold',
        'profit_prior',
        'profit_base',
        'profit_delta_prior_minus_base',
        'charge_start_prior',
        'discharge_start_prior',
        'charge_start_base',
        'discharge_start_base',
        'window_changed',
    ]
    lines = [
        '# Current Champion Daily Error Review',
        '',
        'Champion reviewed: `ens_champion_segmented6_prior`.',
        '',
        '## Overall',
        '',
        f'- Days: `{len(df)}`.',
        f'- Day-weighted mean profit: `{df["profit"].mean():.4f}`.',
        '- Note: official report tables use fold-mean profit; day-weighted mean differs '
        'because folds have different day counts.',
        f'- Mean oracle profit: `{df["oracle_profit"].mean():.4f}`.',
        f'- Mean oracle ratio: `{(df["profit"].sum() / df["oracle_profit"].sum()):.4f}`.',
        f'- Mean regret: `{df["regret"].mean():.4f}`.',
        f'- Loss days: `{int(df["loss_day"].sum())}`.',
        f'- Exact charge hit rate: `{(df["charge_start_gap"] == 0).mean():.4f}`.',
        f'- Exact discharge hit rate: `{(df["discharge_start_gap"] == 0).mean():.4f}`.',
        f'- Both windows within 2 slots: `{df["within_2_slots"].mean():.4f}`.',
        '',
        '## Fold Summary',
        '',
        markdown_table(fsum),
        '',
        '## Gap Direction Summary',
        '',
        markdown_table(gsum),
        '',
        '## Loss Days',
        '',
        markdown_table(loss_days[keep_day]),
        '',
        '## Highest-Regret Days',
        '',
        markdown_table(high_regret[keep_day]),
        '',
        '## Lowest Oracle-Ratio Days',
        '',
        markdown_table(low_ratio[keep_day]),
        '',
        '## Dispatch Prior Delta vs Prior-Disabled Champion',
        '',
        markdown_table(prior_sum),
        '',
        '### Biggest Prior Improvements',
        '',
        markdown_table(top_gain[delta_keep]),
        '',
        '### Biggest Prior Regressions',
        '',
        markdown_table(top_loss[delta_keep]),
        '',
        '## Bid-Space Diagnostic Correlation',
        '',
        markdown_table(bid_corr),
        '',
        '## Readout',
        '',
        '- The hardest fold is October: highest mean regret and the worst oracle ratio.',
        '- Discharge timing is now often the larger error after adding prior, '
        'especially in October and December.',
        '- Prior helps in aggregate, but it changes only a subset of days; '
        'the next step should inspect changed days rather than retune globally.',
        '- High regret days are usually not low-spread/no-trade problems; '
        'they are wrong-window problems.',
        '- Bid-space diagnostics should be used to explain candidate-window mistakes, '
        'not to directly replace the price model.',
    ]
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUT_MD)
    print(fsum.to_string(index=False))
    print(gsum.to_string(index=False))
    print(loss_days[keep_day].to_string(index=False))


if __name__ == '__main__':
    write_report()
