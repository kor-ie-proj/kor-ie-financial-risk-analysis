'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import {
    ApiError,
    fetchCompanies,
    fetchFinancials,
    fetchIndicators,
    fetchManualRisk,
    fetchRisk,
    type CompanyListResponse,
    type FinancialResponse,
    type IndicatorResponse,
    type ManualIndicatorAdjustments,
    type ManualRiskRequest,
    type RiskResponse,
} from '@/lib/api';
import { cn } from '@/lib/utils';
import {
    Building2,
    LineChart as LineChartIcon,
    ShieldCheck,
    TrendingUp,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

type FetchState<T> = {
    data: T | null;
    loading: boolean;
    error: string | null;
};

const palette = [
    'var(--chart-1)',
    'var(--chart-2)',
    'var(--chart-3)',
    'var(--chart-4)',
    'var(--chart-5)',
];

const cardSurface =
    'border border-border/60 bg-card/95 shadow-sm backdrop-blur-sm transition-shadow hover:shadow-lg';
const indicatorPillClass =
    'rounded-full border border-border/60 bg-muted/40 px-3 py-1 text-xs uppercase tracking-wide text-muted-foreground backdrop-blur-sm';

const MANUAL_ADJUSTMENT_DEFAULTS: ManualIndicatorAdjustments = {
    construction_bsi_actual: 0,
    base_rate: 0,
    housing_sale_price: 0,
    m2_growth: 0,
};

const manualAdjustmentLabels: Record<keyof ManualIndicatorAdjustments, string> =
    {
        construction_bsi_actual: 'Construction BSI (actual)',
        base_rate: 'Base rate',
        housing_sale_price: 'Housing sale price index',
        m2_growth: 'M2 growth',
    };

const manualAdjustmentHelp: Record<keyof ManualIndicatorAdjustments, string> = {
    construction_bsi_actual: 'Percent change vs latest quarter',
    base_rate: 'Change in percentage points',
    housing_sale_price: 'Percent change vs latest quarter',
    m2_growth: 'Percent change vs latest quarter',
};

const metricPriority = [
    'revenue',
    'operating_profit',
    'quarterly_profit',
    'debt_ratio',
    'roe',
    'roa',
];

const riskTone: Record<RiskResponse['risk_level'], string> = {
    Low: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    Moderate: 'bg-amber-100 text-amber-900 border-amber-200',
    High: 'bg-rose-100 text-rose-900 border-rose-200',
    Critical: 'bg-red-100 text-red-900 border-red-200',
};

const riskCopy: Record<RiskResponse['risk_level'], string> = {
    Low: 'Solid fundamentals with limited near-term stress.',
    Moderate: 'Monitor liquidity and macro exposure closely.',
    High: 'Elevated risk detected. Consider mitigation plans.',
    Critical: 'Immediate action required. High default risk.',
};

const compactCurrency = new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
});

const percentFormatter = new Intl.NumberFormat('en-US', {
    style: 'percent',
    maximumFractionDigits: 1,
});

const decimalFormatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 2,
});

const indicatorDateFormatter = new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
});

type IndicatorRange = '12M' | '24M' | '60M' | 'ALL';

const indicatorRangeOptions: { value: IndicatorRange; label: string }[] = [
    { value: '12M', label: '12 months' },
    { value: '24M', label: '24 months' },
    { value: '60M', label: '5 years' },
    { value: 'ALL', label: 'All data' },
];

function formatMetricValue(metric: string, value: number): string {
    const lowerKey = metric.toLowerCase();
    if (Number.isNaN(value)) {
        return '-';
    }

    if (
        lowerKey.includes('ratio') ||
        lowerKey.includes('growth') ||
        lowerKey === 'roe' ||
        lowerKey === 'roa'
    ) {
        return percentFormatter.format(value / 100);
    }

    return compactCurrency.format(value);
}

function formatIndicatorLabel(column: string): string {
    return column
        .split('_')
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function formatIndicatorValue(value: number): string {
    if (Number.isNaN(value)) {
        return '-';
    }
    return decimalFormatter.format(value);
}

function formatIndicatorDate(value: string): string {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    return indicatorDateFormatter.format(date);
}

function formatMetricTick(metric: string | null, value: number): string {
    if (Number.isNaN(value)) return '-';
    if (!metric) return decimalFormatter.format(value);
    return formatMetricValue(metric, value);
}

function buildFinancialChartData(
    source?: FinancialResponse | null,
    metric?: string
) {
    if (!source || !metric) return [];
    return source.data.map((row) => ({
        period: row.period,
        value: row.metrics[metric],
    }));
}

function getSparseXAxisInterval(length: number, desiredTicks = 6) {
    if (length <= desiredTicks) return 0;
    return Math.max(1, Math.floor(length / desiredTicks));
}

function getIndicatorSlice(range: IndicatorRange, total: number) {
    if (range === 'ALL') {
        return total;
    }
    const months = parseInt(range.replace('M', ''), 10);
    if (Number.isNaN(months)) {
        return total;
    }
    return Math.min(total, months);
}

function useAnimatedNumber(target: number | null | undefined, duration = 700) {
    const [displayValue, setDisplayValue] = useState(() => target ?? 0);
    const frameRef = useRef<number | undefined>(undefined);
    const startRef = useRef<number | undefined>(undefined);
    const fromRef = useRef<number>(target ?? 0);

    useEffect(() => {
        if (target === null || target === undefined || Number.isNaN(target)) {
            setDisplayValue(0);
            fromRef.current = 0;
            return () => {
                if (frameRef.current) cancelAnimationFrame(frameRef.current);
            };
        }

        const fromValue = fromRef.current;
        const toValue = target;
        if (Math.abs(toValue - fromValue) < 0.001) {
            setDisplayValue(toValue);
            fromRef.current = toValue;
            return () => {
                if (frameRef.current) cancelAnimationFrame(frameRef.current);
            };
        }

        if (frameRef.current) cancelAnimationFrame(frameRef.current);
        startRef.current = undefined;

        const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

        function step(timestamp: number) {
            if (startRef.current === undefined) {
                startRef.current = timestamp;
            }
            const progress = Math.min(
                (timestamp - startRef.current) / duration,
                1
            );
            const eased = easeOut(progress);
            const next = fromValue + (toValue - fromValue) * eased;
            setDisplayValue(next);
            if (progress < 1) {
                frameRef.current = requestAnimationFrame(step);
            } else {
                fromRef.current = toValue;
            }
        }

        frameRef.current = requestAnimationFrame(step);

        return () => {
            if (frameRef.current) cancelAnimationFrame(frameRef.current);
        };
    }, [target, duration]);

    return displayValue;
}

function Home() {
    const [indicatorState, setIndicatorState] = useState<
        FetchState<IndicatorResponse>
    >({
        data: null,
        loading: true,
        error: null,
    });
    const [companyState, setCompanyState] = useState<
        FetchState<CompanyListResponse>
    >({
        data: null,
        loading: true,
        error: null,
    });
    const [riskState, setRiskState] = useState<FetchState<RiskResponse>>({
        data: null,
        loading: false,
        error: null,
    });
    const [financialState, setFinancialState] = useState<
        FetchState<FinancialResponse>
    >({
        data: null,
        loading: false,
        error: null,
    });

    const [searchTerm, setSearchTerm] = useState('');
    const [selectedCompany, setSelectedCompany] = useState<string | null>(null);
    const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
    const [indicatorRange, setIndicatorRange] = useState<IndicatorRange>('ALL');
    const [riskMode, setRiskMode] = useState<'forecast' | 'manual'>('forecast');
    const [manualRiskState, setManualRiskState] = useState<
        FetchState<RiskResponse>
    >({
        data: null,
        loading: false,
        error: null,
    });

    useEffect(() => {
        let cancelled = false;
        async function loadIndicators() {
            try {
                const data = await fetchIndicators({
                    limit: 120,
                    columns: 'all',
                });
                if (cancelled) return;
                setIndicatorState({ data, loading: false, error: null });
            } catch (error) {
                if (cancelled) return;
                setIndicatorState({
                    data: null,
                    loading: false,
                    error:
                        error instanceof Error
                            ? error.message
                            : 'Failed to load indicators',
                });
            }
        }
        loadIndicators();
        return () => {
            cancelled = true;
        };
    }, []);

    useEffect(() => {
        setManualRiskState({ data: null, loading: false, error: null });
        setManualAdjustments({ ...MANUAL_ADJUSTMENT_DEFAULTS });
        setRiskMode('forecast');
    }, [selectedCompany]);

    useEffect(() => {
        let cancelled = false;
        async function loadCompanies(term?: string) {
            try {
                setCompanyState((prev) => ({ ...prev, loading: true }));
                const data = await fetchCompanies({ limit: 100, q: term });
                if (cancelled) return;
                setCompanyState({ data, loading: false, error: null });
                if (!selectedCompany && data.companies.length > 0) {
                    setSelectedCompany(data.companies[0]);
                }
            } catch (error) {
                if (cancelled) return;
                setCompanyState({
                    data: null,
                    loading: false,
                    error:
                        error instanceof Error
                            ? error.message
                            : 'Failed to load companies',
                });
            }
        }

        const timeout = setTimeout(() => {
            loadCompanies(searchTerm || undefined);
        }, 300);

        return () => {
            cancelled = true;
            clearTimeout(timeout);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchTerm]);

    useEffect(() => {
        if (!selectedCompany) return;

        const company = selectedCompany;
        let cancelled = false;
        async function loadRisk() {
            setRiskState({ data: null, loading: true, error: null });
            try {
                const data = await fetchRisk(company);
                if (cancelled) return;
                setRiskState({ data, loading: false, error: null });
            } catch (error) {
                if (cancelled) return;
                setRiskState({
                    data: null,
                    loading: false,
                    error:
                        error instanceof Error
                            ? error.message
                            : 'Failed to load risk',
                });
            }
        }

        async function loadFinancials() {
            setFinancialState({ data: null, loading: true, error: null });
            try {
                const data = await fetchFinancials(company);
                if (cancelled) return;
                setFinancialState({ data, loading: false, error: null });
                const defaultMetric =
                    metricPriority.find((metric) =>
                        data.available_metrics.includes(metric)
                    ) ||
                    data.available_metrics[0] ||
                    null;
                setSelectedMetric((current) =>
                    current && data.available_metrics.includes(current)
                        ? current
                        : defaultMetric
                );
            } catch (error) {
                if (cancelled) return;
                setFinancialState({
                    data: null,
                    loading: false,
                    error:
                        error instanceof Error
                            ? error.message
                            : 'Failed to load financials',
                });
                setSelectedMetric(null);
            }
        }

        loadRisk();
        loadFinancials();

        return () => {
            cancelled = true;
        };
    }, [selectedCompany]);

    const indicatorSeries = useMemo(() => {
        const indicatorData = indicatorState.data;
        if (!indicatorData) return [];
        const sliceSize = getIndicatorSlice(
            indicatorRange,
            indicatorData.data.length
        );
        const scopedRows =
            sliceSize >= indicatorData.data.length
                ? indicatorData.data
                : indicatorData.data.slice(-sliceSize);

        return indicatorData.columns.map((column, index) => ({
            key: column,
            label: formatIndicatorLabel(column),
            color: palette[index % palette.length],
            data: scopedRows.map((row) => ({
                date: row.date,
                value: row.values[column] ?? null,
            })),
        }));
    }, [indicatorState.data, indicatorRange]);
    const financialChartData = useMemo(
        () =>
            buildFinancialChartData(
                financialState.data,
                selectedMetric || undefined
            ),
        [financialState.data, selectedMetric]
    );

    const availableMetrics = financialState.data?.available_metrics ?? [];
    const activeRiskState = riskMode === 'manual' ? manualRiskState : riskState;
    const activeRisk = activeRiskState.data;
    const activeRiskLoading = activeRiskState.loading;
    const activeRiskError = activeRiskState.error;
    const hasManualResult = Boolean(manualRiskState.data);
    const animatedRiskScore = useAnimatedNumber(activeRisk?.risk_score ?? null);

    const manualAdjustmentEntries = useMemo<
        Array<{ key: keyof ManualIndicatorAdjustments; value: number }>
    >(() => {
        const raw = manualRiskState.data?.manual_adjustments;
        if (!raw) {
            return [] as Array<{
                key: keyof ManualIndicatorAdjustments;
                value: number;
            }>;
        }
        return (
            Object.entries(raw) as Array<
                [keyof ManualIndicatorAdjustments, number]
            >
        )
            .filter(
                ([, value]) => !Number.isNaN(value) && Math.abs(value) > 1e-6
            )
            .map(([key, value]) => ({ key, value }));
    }, [manualRiskState.data?.manual_adjustments]);

    const [manualAdjustments, setManualAdjustments] = useState<
        Record<keyof ManualIndicatorAdjustments, number | string>
    >({
        ...MANUAL_ADJUSTMENT_DEFAULTS,
    });

    const handleManualAdjustmentChange = (
        metric: keyof ManualIndicatorAdjustments,
        rawValue: string
    ) => {
        setManualAdjustments((prev) => ({
            ...prev,
            [metric]: rawValue,
        }));
    };

    const handleManualRiskSubmit = async () => {
        if (!selectedCompany) {
            return;
        }
        setManualRiskState((prev) => ({ ...prev, loading: true, error: null }));
        try {
            const parsedAdjustments = Object.fromEntries(
                Object.entries(manualAdjustments).map(([key, value]) => {
                    const parsed =
                        typeof value === 'number' ? value : parseFloat(value);
                    return [key, isNaN(parsed) ? 0 : parsed];
                })
            ) as ManualIndicatorAdjustments;

            const payload: ManualRiskRequest = {
                adjustments: parsedAdjustments,
            };
            const data = await fetchManualRisk(selectedCompany, payload);
            setManualRiskState({ data, loading: false, error: null });
            setRiskMode('manual');
        } catch (error) {
            const message =
                error instanceof ApiError
                    ? error.message
                    : 'Failed to compute manual risk scenario.';
            setManualRiskState((prev) => ({
                ...prev,
                loading: false,
                error: message,
            }));
        }
    };

    const handleManualReset = () => {
        setManualAdjustments({ ...MANUAL_ADJUSTMENT_DEFAULTS });
        setManualRiskState({ data: null, loading: false, error: null });
        setRiskMode('forecast');
    };

    return (
        <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-muted/40 text-foreground">
            <div
                aria-hidden
                className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,_var(--muted)_0%,_transparent_60%)]"
            />
            <div
                aria-hidden
                className="pointer-events-none absolute inset-y-0 right-0 -z-10 w-1/2 bg-[radial-gradient(circle_at_top_right,_var(--primary)/20,_transparent_65%)]"
            />
            <div className="relative mx-auto flex max-w-7xl flex-col gap-8 px-6 py-10 lg:flex-row lg:items-start">
                <section className="flex w-full flex-col lg:w-2/3">
                    <ScrollArea className="max-h-full lg:h-[calc(100vh-160px)]">
                        <div className="flex flex-col gap-6 pr-2 lg:pr-6">
                            <div className="space-y-2">
                                <div className="flex flex-wrap items-center gap-2 text-primary">
                                    <Building2
                                        className="h-5 w-5"
                                        aria-hidden
                                    />
                                    <h1 className="text-2xl font-semibold leading-tight text-foreground">
                                        Construction Company Risk Analysis
                                    </h1>
                                    <span className={indicatorPillClass}>
                                        Live monitoring
                                    </span>
                                </div>
                                <p className="max-w-2xl text-sm text-muted-foreground">
                                    Compare construction firms, inspect
                                    fundamentals, and track heuristic risk
                                    scores derived from macro forecasts.
                                </p>
                            </div>

                            <Card className={cardSurface}>
                                <CardHeader className="pb-3">
                                    <CardTitle className="text-sm font-medium">
                                        Search Companies
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="pb-4">
                                    <div className="space-y-3">
                                        <div className="space-y-1.5">
                                            <Label htmlFor="company-search">
                                                Company name
                                            </Label>
                                            <Input
                                                id="company-search"
                                                placeholder="Search construction firms"
                                                value={searchTerm}
                                                onChange={(event) =>
                                                    setSearchTerm(
                                                        event.target.value
                                                    )
                                                }
                                            />
                                        </div>
                                        <Separator />
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-xs text-muted-foreground">
                                                <span>Companies</span>
                                                {companyState.loading && (
                                                    <span>Loading…</span>
                                                )}
                                            </div>
                                            <ScrollArea className="h-48 rounded-md border border-border">
                                                <div className="grid gap-1 p-2">
                                                    {companyState.error && (
                                                        <div className="rounded-md bg-amber-100 px-2 py-1 text-xs text-amber-900">
                                                            {companyState.error}
                                                        </div>
                                                    )}
                                                    {companyState.data
                                                        ?.companies.length
                                                        ? companyState.data.companies.map(
                                                              (company) => (
                                                                  <Button
                                                                      key={
                                                                          company
                                                                      }
                                                                      variant={
                                                                          selectedCompany ===
                                                                          company
                                                                              ? 'secondary'
                                                                              : 'ghost'
                                                                      }
                                                                      className="justify-start"
                                                                      onClick={() =>
                                                                          setSelectedCompany(
                                                                              company
                                                                          )
                                                                      }
                                                                  >
                                                                      {company}
                                                                  </Button>
                                                              )
                                                          )
                                                        : !companyState.loading && (
                                                              <span className="px-2 py-3 text-xs text-muted-foreground">
                                                                  No companies
                                                                  found.
                                                              </span>
                                                          )}
                                                </div>
                                            </ScrollArea>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card
                                className={cn(
                                    cardSurface,
                                    'border border-border/70'
                                )}
                            >
                                <CardHeader className="pb-3">
                                    <div className="flex flex-col gap-3">
                                        <div className="flex flex-wrap items-center justify-between gap-3">
                                            <div className="flex items-center gap-2 text-primary">
                                                <ShieldCheck
                                                    className="h-4 w-4"
                                                    aria-hidden
                                                />
                                                <CardTitle className="text-base font-semibold text-foreground">
                                                    Risk Summary
                                                </CardTitle>
                                                {/* <Badge className="border border-border/70 text-xs uppercase tracking-wide text-muted-foreground">
                                                    {riskMode === 'manual' ? 'Manual scenario' : 'Model forecast'}
                                                </Badge> */}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <Button
                                                    size="sm"
                                                    variant={
                                                        riskMode === 'forecast'
                                                            ? 'secondary'
                                                            : 'ghost'
                                                    }
                                                    onClick={() =>
                                                        setRiskMode('forecast')
                                                    }
                                                    disabled={
                                                        !riskState.data ||
                                                        riskState.loading
                                                    }
                                                >
                                                    Model forecast
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant={
                                                        riskMode === 'manual'
                                                            ? 'secondary'
                                                            : 'ghost'
                                                    }
                                                    onClick={() =>
                                                        setRiskMode('manual')
                                                    }
                                                    disabled={
                                                        !manualRiskState.data
                                                    }
                                                >
                                                    Manual scenario
                                                </Button>
                                            </div>
                                        </div>
                                        <p className="text-xs text-muted-foreground">
                                            Adjust the next-quarter macro
                                            indicators to explore what-if risk
                                            scores.
                                        </p>
                                    </div>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    {activeRiskLoading ? (
                                        <Skeleton className="h-24 w-full" />
                                    ) : activeRiskError ? (
                                        <div className="rounded-md bg-amber-100 px-3 py-2 text-sm text-amber-900">
                                            {activeRiskError}
                                        </div>
                                    ) : activeRisk ? (
                                        <div className="space-y-3">
                                            <div className="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <p className="text-xs text-muted-foreground">
                                                        Selected company
                                                    </p>
                                                    <p className="text-base font-medium">
                                                        {activeRisk.corp_name}
                                                    </p>
                                                </div>
                                                <Badge
                                                    className={cn(
                                                        'border text-sm',
                                                        riskTone[
                                                            activeRisk
                                                                .risk_level
                                                        ]
                                                    )}
                                                >
                                                    {activeRisk.risk_level} risk
                                                </Badge>
                                            </div>
                                            {riskMode === 'manual' &&
                                                manualAdjustmentEntries.length >
                                                    0 && (
                                                    <div className="rounded-md border border-dashed border-primary/40 bg-primary/5 px-3 py-2 text-xs text-primary">
                                                        <span className="font-medium">
                                                            Assumptions:
                                                        </span>{' '}
                                                        {manualAdjustmentEntries
                                                            .map(
                                                                ({
                                                                    key,
                                                                    value,
                                                                }) => {
                                                                    const safeValue =
                                                                        typeof value ===
                                                                            'number' &&
                                                                        !isNaN(
                                                                            value
                                                                        )
                                                                            ? value
                                                                            : null;
                                                                    if (
                                                                        key ===
                                                                        'base_rate'
                                                                    ) {
                                                                        return `${
                                                                            manualAdjustmentLabels[
                                                                                key
                                                                            ]
                                                                        } ${
                                                                            safeValue !==
                                                                            null
                                                                                ? (safeValue >=
                                                                                  0
                                                                                      ? '+'
                                                                                      : '') +
                                                                                  safeValue.toFixed(
                                                                                      2
                                                                                  ) +
                                                                                  '%p'
                                                                                : '-'
                                                                        }`;
                                                                    } else {
                                                                        return `${
                                                                            manualAdjustmentLabels[
                                                                                key
                                                                            ]
                                                                        } ${
                                                                            safeValue !==
                                                                            null
                                                                                ? (safeValue >=
                                                                                  0
                                                                                      ? '+'
                                                                                      : '') +
                                                                                  safeValue.toFixed(
                                                                                      1
                                                                                  ) +
                                                                                  '%'
                                                                                : '-'
                                                                        }`;
                                                                    }
                                                                }
                                                            )
                                                            .join(' · ')}
                                                    </div>
                                                )}
                                            <div className="grid gap-2 text-sm">
                                                <div className="flex items-center justify-between rounded-md border border-border px-3 py-2">
                                                    <span className="text-muted-foreground">
                                                        Risk score
                                                    </span>
                                                    <span className="font-semibold">
                                                        {typeof animatedRiskScore ===
                                                            'number' &&
                                                        !isNaN(
                                                            animatedRiskScore
                                                        )
                                                            ? animatedRiskScore.toFixed(
                                                                  1
                                                              )
                                                            : '-'}
                                                    </span>
                                                </div>
                                                <div className="rounded-md bg-secondary px-3 py-2 text-xs text-secondary-foreground">
                                                    {
                                                        riskCopy[
                                                            activeRisk
                                                                .risk_level
                                                        ]
                                                    }
                                                </div>
                                                <div className="text-xs text-muted-foreground">
                                                    Critical threshold ≥{' '}
                                                    {typeof activeRisk
                                                        ?.thresholds?.danger ===
                                                        'number' &&
                                                    !isNaN(
                                                        activeRisk.thresholds
                                                            .danger
                                                    )
                                                        ? activeRisk.thresholds.danger.toFixed(
                                                              0
                                                          )
                                                        : '-'}{' '}
                                                    · High threshold ≥{' '}
                                                    {typeof activeRisk
                                                        ?.thresholds
                                                        ?.caution ===
                                                        'number' &&
                                                    !isNaN(
                                                        activeRisk.thresholds
                                                            .caution
                                                    )
                                                        ? activeRisk.thresholds.caution.toFixed(
                                                              0
                                                          )
                                                        : '-'}
                                                    · Moderate threshold ≥{' '}
                                                    {typeof activeRisk
                                                        ?.thresholds?.safe ===
                                                        'number' &&
                                                    !isNaN(
                                                        activeRisk.thresholds
                                                            .safe
                                                    )
                                                        ? activeRisk.thresholds.safe.toFixed(
                                                              0
                                                          )
                                                        : '-'}
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <p className="text-sm text-muted-foreground">
                                            Select a company to view its risk
                                            profile.
                                        </p>
                                    )}
                                    <Separator />
                                    <div className="space-y-3">
                                        <div>
                                            <p className="text-sm font-medium text-foreground">
                                                Manual scenario inputs
                                            </p>
                                            <p className="text-xs text-muted-foreground">
                                                Provide expected changes
                                                relative to the latest quarter
                                                averages.
                                            </p>
                                        </div>
                                        <div className="grid gap-3 sm:grid-cols-2">
                                            {(
                                                Object.keys(
                                                    manualAdjustments
                                                ) as Array<
                                                    keyof ManualIndicatorAdjustments
                                                >
                                            ).map((key) => (
                                                <div
                                                    key={key}
                                                    className="space-y-1.5"
                                                >
                                                    <Label
                                                        htmlFor={`manual-${key}`}
                                                    >
                                                        {
                                                            manualAdjustmentLabels[
                                                                key
                                                            ]
                                                        }
                                                    </Label>
                                                    <div className="flex items-center gap-2">
                                                        <Input
                                                            id={`manual-${key}`}
                                                            type="number"
                                                            inputMode="decimal"
                                                            step={
                                                                key ===
                                                                'base_rate'
                                                                    ? 0.01
                                                                    : 0.1
                                                            }
                                                            value={
                                                                manualAdjustments[
                                                                    key
                                                                ]
                                                            }
                                                            onChange={(event) =>
                                                                handleManualAdjustmentChange(
                                                                    key,
                                                                    event.target
                                                                        .value
                                                                )
                                                            }
                                                            disabled={
                                                                !selectedCompany ||
                                                                riskState.loading
                                                            }
                                                        />
                                                        <span className="text-xs text-muted-foreground">
                                                            {
                                                                manualAdjustmentHelp[
                                                                    key
                                                                ]
                                                            }
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        {manualRiskState.error && (
                                            <div className="rounded-md bg-amber-100 px-3 py-2 text-xs text-amber-900">
                                                {manualRiskState.error}
                                            </div>
                                        )}
                                        <div className="flex flex-wrap items-center gap-2">
                                            <Button
                                                size="sm"
                                                onClick={handleManualRiskSubmit}
                                                disabled={
                                                    !selectedCompany ||
                                                    riskState.loading ||
                                                    manualRiskState.loading
                                                }
                                            >
                                                {manualRiskState.loading
                                                    ? 'Calculating…'
                                                    : 'Calculate scenario'}
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={handleManualReset}
                                                disabled={
                                                    !hasManualResult &&
                                                    !manualRiskState.loading
                                                }
                                            >
                                                Reset
                                            </Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className={cn(cardSurface, 'flex-1')}>
                                <CardHeader className="pb-3">
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center gap-2 text-primary">
                                            <LineChartIcon
                                                className="h-4 w-4"
                                                aria-hidden
                                            />
                                            <CardTitle className="text-base font-semibold text-foreground">
                                                Financial Trajectory
                                            </CardTitle>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className="flex-1">
                                                <p className="text-xs text-muted-foreground">
                                                    Quarterly data sourced from
                                                    DART filings
                                                </p>
                                            </div>
                                            <Select
                                                value={
                                                    selectedMetric || undefined
                                                }
                                                onValueChange={
                                                    setSelectedMetric
                                                }
                                                disabled={
                                                    availableMetrics.length ===
                                                    0
                                                }
                                            >
                                                <SelectTrigger className="h-9 w-[200px] text-sm">
                                                    <SelectValue placeholder="Choose metric" />
                                                </SelectTrigger>
                                                <SelectContent>
                                                    {availableMetrics.map(
                                                        (metric) => (
                                                            <SelectItem
                                                                key={metric}
                                                                value={metric}
                                                            >
                                                                {metric.replaceAll(
                                                                    '_',
                                                                    ' '
                                                                )}
                                                            </SelectItem>
                                                        )
                                                    )}
                                                </SelectContent>
                                            </Select>
                                        </div>
                                    </div>
                                </CardHeader>
                                <CardContent className="h-[320px]">
                                    {financialState.loading ? (
                                        <div className="flex h-full items-center justify-center">
                                            <Skeleton className="h-40 w-full" />
                                        </div>
                                    ) : financialState.error ? (
                                        <div className="flex h-full items-center justify-center text-sm text-destructive">
                                            {financialState.error}
                                        </div>
                                    ) : !selectedMetric ||
                                      financialChartData.length === 0 ? (
                                        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                                            {selectedCompany
                                                ? 'No financial data for this metric.'
                                                : 'Select a company to view financial trends.'}
                                        </div>
                                    ) : (
                                        <ResponsiveContainer
                                            width="100%"
                                            height="100%"
                                        >
                                            <LineChart
                                                key={`financial-${
                                                    selectedCompany ?? 'none'
                                                }-${selectedMetric ?? 'none'}-${
                                                    financialChartData.length
                                                }`}
                                                data={financialChartData}
                                                margin={{
                                                    top: 10,
                                                    right: 16,
                                                    left: 0,
                                                    bottom: 10,
                                                }}
                                            >
                                                <CartesianGrid
                                                    strokeDasharray="3 3"
                                                    stroke="var(--border)"
                                                />
                                                <XAxis
                                                    dataKey="period"
                                                    stroke="var(--muted-foreground)"
                                                    tickLine={false}
                                                    interval={getSparseXAxisInterval(
                                                        financialChartData.length
                                                    )}
                                                    tick={false}
                                                    axisLine={false}
                                                />
                                                <YAxis
                                                    stroke="var(--muted-foreground)"
                                                    tickLine={false}
                                                    tick={{ fontSize: 12 }}
                                                    width={70}
                                                    tickFormatter={(
                                                        value: number
                                                    ) =>
                                                        formatMetricTick(
                                                            selectedMetric,
                                                            value
                                                        )
                                                    }
                                                />
                                                <Tooltip
                                                    formatter={(
                                                        value: number
                                                    ) => [
                                                        formatMetricValue(
                                                            selectedMetric,
                                                            value
                                                        ),
                                                        selectedMetric?.replaceAll(
                                                            '_',
                                                            ' '
                                                        ) || '',
                                                    ]}
                                                    labelFormatter={(value) =>
                                                        value
                                                    }
                                                    contentStyle={{
                                                        background:
                                                            'var(--card)',
                                                        borderRadius: 12,
                                                        border: '1px solid var(--border)',
                                                        color: 'var(--card-foreground)',
                                                    }}
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="value"
                                                    stroke="var(--chart-1)"
                                                    strokeWidth={2}
                                                    dot={false}
                                                    isAnimationActive
                                                    animationBegin={100}
                                                    animationDuration={900}
                                                    animationEasing="ease-in-out"
                                                    animateNewValues={false}
                                                />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    )}
                                </CardContent>
                            </Card>
                        </div>
                    </ScrollArea>
                </section>

                <section className="flex w-full flex-col lg:w-1/3">
                    <ScrollArea className="max-h-full lg:h-[calc(100vh-160px)]">
                        <div className="flex flex-col gap-4 pr-2">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="flex flex-wrap items-center gap-2 text-primary">
                                    <TrendingUp
                                        className="h-5 w-5"
                                        aria-hidden
                                    />
                                    <h2 className="text-xl font-semibold leading-tight text-foreground">
                                        Macro Indicators
                                    </h2>
                                    <span className={indicatorPillClass}>
                                        Monthly frequency
                                    </span>
                                </div>
                                <div className="flex items-center gap-2 text-xs">
                                    <Select
                                        value={indicatorRange}
                                        onValueChange={(value) =>
                                            setIndicatorRange(
                                                value as IndicatorRange
                                            )
                                        }
                                    >
                                        <SelectTrigger className="h-9 w-[180px] text-xs">
                                            <SelectValue placeholder="Date range" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {indicatorRangeOptions.map(
                                                (option) => (
                                                    <SelectItem
                                                        key={option.value}
                                                        value={option.value}
                                                    >
                                                        {option.label}
                                                    </SelectItem>
                                                )
                                            )}
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                            {indicatorState.loading ? (
                                <div className="flex h-[420px] items-center justify-center">
                                    <Skeleton className="h-40 w-full" />
                                </div>
                            ) : indicatorState.error ? (
                                <div className="flex h-[420px] items-center justify-center rounded-md border border-dashed border-border text-sm text-destructive">
                                    {indicatorState.error}
                                </div>
                            ) : indicatorSeries.length === 0 ? (
                                <div className="flex h-[420px] items-center justify-center rounded-md border border-dashed border-border text-sm text-muted-foreground">
                                    No indicator data available.
                                </div>
                            ) : (
                                <div className="space-y-3 pb-2">
                                    {indicatorSeries.map((series) => (
                                        <Card
                                            key={series.key}
                                            className={cn(
                                                cardSurface,
                                                'min-h-[220px]'
                                            )}
                                        >
                                            <CardHeader className="pb-3">
                                                <CardTitle className="text-sm font-medium">
                                                    {series.label}
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent className="h-[180px]">
                                                <ResponsiveContainer
                                                    width="100%"
                                                    height="100%"
                                                >
                                                    <LineChart
                                                        key={`${
                                                            series.key
                                                        }-${indicatorRange}-${
                                                            series.data.length
                                                                ? series.data[
                                                                      series
                                                                          .data
                                                                          .length -
                                                                          1
                                                                  ].date
                                                                : 'empty'
                                                        }`}
                                                        data={series.data}
                                                        margin={{
                                                            top: 10,
                                                            right: 16,
                                                            left: 0,
                                                            bottom: 10,
                                                        }}
                                                    >
                                                        <CartesianGrid
                                                            strokeDasharray="3 3"
                                                            stroke="var(--border)"
                                                        />
                                                        <XAxis
                                                            dataKey="date"
                                                            stroke="var(--muted-foreground)"
                                                            tickLine={false}
                                                            interval={getSparseXAxisInterval(
                                                                series.data
                                                                    .length,
                                                                5
                                                            )}
                                                            tick={false}
                                                            axisLine={false}
                                                        />
                                                        <YAxis
                                                            stroke="var(--muted-foreground)"
                                                            tickLine={false}
                                                            tick={{
                                                                fontSize: 12,
                                                            }}
                                                            width={60}
                                                            tickFormatter={(
                                                                value: number
                                                            ) =>
                                                                formatIndicatorValue(
                                                                    value
                                                                )
                                                            }
                                                        />
                                                        <Tooltip
                                                            formatter={(
                                                                value: number
                                                            ) => [
                                                                formatIndicatorValue(
                                                                    value
                                                                ),
                                                                series.label,
                                                            ]}
                                                            labelFormatter={(
                                                                value
                                                            ) =>
                                                                formatIndicatorDate(
                                                                    String(
                                                                        value
                                                                    )
                                                                )
                                                            }
                                                            contentStyle={{
                                                                background:
                                                                    'var(--card)',
                                                                borderRadius: 12,
                                                                border: '1px solid var(--border)',
                                                                color: 'var(--card-foreground)',
                                                            }}
                                                        />
                                                        <Line
                                                            type="monotone"
                                                            dataKey="value"
                                                            stroke={
                                                                series.color
                                                            }
                                                            strokeWidth={2}
                                                            dot={false}
                                                            isAnimationActive
                                                            animationBegin={100}
                                                            animationDuration={
                                                                900
                                                            }
                                                            animationEasing="ease-in-out"
                                                            animateNewValues={
                                                                false
                                                            }
                                                        />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            )}
                        </div>
                    </ScrollArea>
                </section>
            </div>
        </div>
    );
}

export default Home;
