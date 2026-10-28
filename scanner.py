"""
BUSCADOR DE OPORTUNIDADES
=========================
Baja datos diarios de unas 500 acciones (S&P 500) y unas 60 criptos, calcula RSI, tendencia, volumen...
y escribe docs/datos.json. La pagina docs/index.html lo muestra como una tabla con filtros.
No opera ni manda avisos: es una lista para mirar en el ordenador.
"""
import os, io, re, sys, json, time, datetime
import numpy as np, pandas as pd, requests

OUT = os.environ.get("OUT", "docs/datos.json")
MIN_DOLARES_ACCION = 5e6      # se ignoran las acciones con menos de 5 M$ de volumen diario medio
STOP_ATR = 2.5
PRECIO_MAX = 200.0            # solo activos con precio inferior a esto (dolares)
RSI_BAJO, RSI_ALTO = 30, 70   # sobrevendida / sobrecomprada

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","AVAX-USD","TRX-USD","LINK-USD",
  "DOT-USD","LTC-USD","NEAR-USD","ATOM-USD","XLM-USD","BCH-USD","ETC-USD","HBAR-USD","ALGO-USD","VET-USD","FIL-USD",
  "ICP-USD","AAVE-USD","UNI7083-USD","MKR-USD","XMR-USD","EOS-USD","XTZ-USD","THETA-USD","SAND-USD","MANA-USD",
  "AXS-USD","GRT6719-USD","EGLD-USD","FLOW-USD","CHZ-USD","ZEC-USD","DASH-USD","KAVA-USD","CRV-USD","COMP-USD",
  "SNX-USD","1INCH-USD","ENJ-USD","BAT-USD","ZIL-USD","IOTA-USD","NEO-USD","QTUM-USD","RUNE-USD","INJ-USD",
  "RENDER-USD","SEI-USD","STX4847-USD","APT21794-USD","ARB11841-USD","OP-USD","SUI20947-USD","LDO-USD","MATIC-USD","POL-USD"]
EXTRA_ACCIONES = ["MRNA","NVO","ETSY","ENPH","SIRI","OKLO","RIVN","RNG","LVS","HOOD","IONQ","SOFI","MARA","PLTR","COIN","AAL","CCL",
  "VALE","GOLD","SLV","GDX","SNAP","LYFT","PATH","AI","GLD","QQQ","IWM","TLT","SPY"]
FALLBACK_ACCIONES = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","JPM","V","MA","UNH","XOM","LLY","COST","HD","PG",
  "NFLX","AMD","CRM","ORCL","ADBE","INTC","BAC","WFC","KO","PEP","DIS","NKE","UBER","PYPL","F","T","PFE"]

def nombre(t): return re.sub(r"\d*-USD$", "", t)

def rsi_w(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
def atr_w(h, l, c, n=14):
    pc = c.shift(1); return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).ewm(alpha=1/n, adjust=False).mean()

def fundamentales(yahoo):
    """Valor 'intrinseco' orientativo de una accion: objetivo medio de analistas y numero de Graham."""
    import yfinance as yf
    info = None
    for _ in range(2):
        try: info = yf.Ticker(yahoo).info; break
        except Exception: time.sleep(2)
    if not info: return {}
    eps, bv = info.get("trailingEps"), info.get("bookValue")
    graham = float((22.5 * eps * bv) ** 0.5) if (eps and bv and eps > 0 and bv > 0) else None
    tgt = info.get("targetMeanPrice")
    return dict(objetivo=float(tgt) if tgt else None, n_analistas=info.get("numberOfAnalystOpinions"), graham=graham,
                per=info.get("trailingPE"), per_futuro=info.get("forwardPE"), recomendacion=info.get("recommendationKey"),
                eps=eps, valor_libro=bv, nombre_largo=info.get("shortName"), sector=info.get("sector"))

def lista_sp500():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        t = pd.read_html(io.StringIO(r.text))[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if len(t) > 400: return t
    except Exception as e: print("  (no pude bajar la lista del S&P 500:", e, ")")
    return FALLBACK_ACCIONES

def bajar(tickers):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 40):
        lote = tickers[i:i + 40]
        try: df = yf.download(lote, period="2y", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
        except Exception as e: print("  lote fallido:", e); continue
        for t in lote:
            try:
                x = df[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
                if x.index.tz is not None: x.index = x.index.tz_localize(None)
                if len(x) >= 230: out[t] = x
            except Exception: pass
        print(f"  descargados {len(out)} de {i + len(lote)}", flush=True); time.sleep(1)
    return out

def analizar(x, tipo, ahora):
    ult_parcial = x.index[-1].normalize() >= ahora.normalize() and (tipo == "cripto" or (ahora.hour, ahora.minute) < (21, 30))
    cerr = x.iloc[:-1] if ult_parcial else x
    if len(cerr) < 210: return None
    c = cerr.Close; e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]; e200 = c.ewm(span=200, adjust=False).mean().iloc[-1]
    e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    atr = float(atr_w(cerr.High, cerr.Low, c).iloc[-1]); pc = float(c.iloc[-1]); px = float(x.Close.iloc[-1])
    vm = float(cerr.Volume.iloc[-21:-1].mean()); dv = float((c * cerr.Volume).iloc[-20:].mean())
    if tipo == "accion" and (px < 1 or dv < MIN_DOLARES_ACCION): return None
    rsi_c = float(rsi_w(c).iloc[-1]); rsi_a = float(rsi_w(x.Close).iloc[-1])
    tend = "alcista" if (pc > e200 and e50 > e200) else ("bajista" if (pc < e200 and e50 < e200) else "mixta")
    return dict(precio=px, cambio_dia=(px / pc - 1) * 100 if ult_parcial else (pc / float(c.iloc[-2]) - 1) * 100,
                rsi=rsi_c, rsi_ahora=rsi_a, tendencia=tend, ret90=(pc / float(c.iloc[-91]) - 1) * 100,
                vol_rel=float(cerr.Volume.iloc[-1] / vm) if vm > 0 else 1.0,
                dist_max20=(px / float(x.High.iloc[-20:].max()) - 1) * 100, dist_ema20=(px / e20 - 1) * 100,
                atr_pct=atr / px * 100, stop=px - STOP_ATR * atr, stop_pct=STOP_ATR * atr / px * 100,
                dolares_dia=dv, ultima_vela=str(cerr.index[-1].date()))

def main():
    ahora = pd.Timestamp.utcnow().tz_localize(None)
    acc_t = sorted(set(lista_sp500() + EXTRA_ACCIONES))
    datos = bajar(CRYPTO + acc_t)
    activos = []; mercado = {}
    for t, x in datos.items():
        tipo = "cripto" if t.endswith("-USD") else "accion"
        try: r = analizar(x, tipo, ahora)
        except Exception as e: print("  error", t, e); continue
        if r is None: continue
        r["t"] = nombre(t); r["yahoo"] = t; r["tipo"] = tipo; activos.append(r)
        if t in ("BTC-USD", "SPY"): mercado[t] = dict(tendencia=r["tendencia"], rsi=round(r["rsi"], 1), precio=r["precio"])
    if not activos: print("✗ sin datos"); sys.exit(1)
    # solo se muestran las sobrevendidas / sobrecompradas con precio menor que PRECIO_MAX
    sel = []
    for a in activos:
        if a["precio"] >= PRECIO_MAX: continue
        r = min(a["rsi"], a["rsi_ahora"]) if (a["rsi"] <= RSI_BAJO or a["rsi_ahora"] <= RSI_BAJO) else max(a["rsi"], a["rsi_ahora"])
        if r <= RSI_BAJO: a["estado"] = "sobrevendida"
        elif r >= RSI_ALTO: a["estado"] = "sobrecomprada"
        else: continue
        sel.append(a)
    print(f"  {len(sel)} activos en sobreventa/sobrecompra y por debajo de {PRECIO_MAX:.0f} $")
    for a in sel:
        if a["tipo"] == "accion":
            a.update({k: v for k, v in fundamentales(a["yahoo"]).items() if v is not None}); time.sleep(0.3)
    activos = sel
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(actualizado=ahora.strftime("%Y-%m-%d %H:%M UTC"), mercado=mercado, total_analizados=len(datos), precio_max=PRECIO_MAX,
                       activos=sorted(activos, key=lambda a: a["rsi"])), f, ensure_ascii=False, separators=(",", ":"))
    print(f"✓ {len(activos)} activos escritos en {OUT} (de {len(datos)} analizados)")

if __name__ == "__main__":
    main()
