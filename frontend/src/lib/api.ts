const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8003").replace(/\/+$/, "");

type RequestParams = Record<string, string | number | undefined>;

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function buildUrl(path: string, params?: RequestParams) {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url.toString();
}

async function getJson<T>(path: string, params?: RequestParams): Promise<T> {
  const url = buildUrl(path, params);
  const res = await fetch(url, {
    headers: {
      Accept: "application/json",
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new ApiError(detail || res.statusText, res.status);
  }

  return (await res.json()) as T;
}

export type IndicatorRecord = {
  date: string;
  values: Record<string, number>;
};

export type IndicatorResponse = {
  columns: string[];
  data: IndicatorRecord[];
};

export type CompanyListResponse = {
  companies: string[];
};

export type FinancialRecord = {
  period: string;
  year: number;
  quarter: string;
  metrics: Record<string, number>;
};

export type FinancialResponse = {
  corp_name: string;
  available_metrics: string[];
  data: FinancialRecord[];
};

export type RiskResponse = {
  corp_name: string;
  risk_score: number;
  normalized_score: number;
  risk_level: "Low" | "Moderate" | "High";
  thresholds: {
    medium: number;
    high: number;
  };
  components: Record<string, number>;
  ecos_quarters: Record<string, unknown>;
  dart_vector: Record<string, number>;
};

export async function fetchIndicators(params?: RequestParams) {
  return getJson<IndicatorResponse>("/indicators", params);
}

export async function fetchCompanies(params?: RequestParams) {
  return getJson<CompanyListResponse>("/companies", params);
}

export async function fetchFinancials(corpName: string) {
  return getJson<FinancialResponse>(`/companies/${encodeURIComponent(corpName)}/financials`);
}

export async function fetchRisk(corpName: string, monthsToPredict = 3) {
  return getJson<RiskResponse>(`/companies/${encodeURIComponent(corpName)}/risk`, {
    months_to_predict: monthsToPredict,
  });
}

export { ApiError };
