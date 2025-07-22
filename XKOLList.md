# Xinfluencer KOL Lists

**Central reference for all Key Opinion Leader (KOL) accounts used in the Xinfluencer AI system.**

---

## 🎯 **PRIMARY KOL LIST: Focused Crypto/RWA Experts**
*Quality over quantity - Technical builders and educators*

**Source:** `focused_kol_list.py` - **RECOMMENDED FOR BULLETPROOF COLLECTION**

### **Core Ethereum/DeFi Builders (Highest Signal)**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@VitalikButerin** | Ethereum founder | Technical insights, protocol development | ⭐⭐⭐⭐⭐ |
| **@haydenzadams** | Uniswap founder | DeFi innovation, AMM protocols | ⭐⭐⭐⭐⭐ |
| **@stani_kulechov** | Aave founder | Lending protocols, DeFi infrastructure | ⭐⭐⭐⭐⭐ |

### **DeFi Protocol Architects**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@AndreCronjeTech** | DeFi architect | YFI, Fantom, protocol design | ⭐⭐⭐⭐ |
| **@rleshner** | Compound founder | Money markets, lending | ⭐⭐⭐⭐ |
| **@bantg** | Yearn core developer | Yield strategies, protocol development | ⭐⭐⭐⭐ |

### **RWA/Institutional Crypto**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@centrifuge** | RWA protocol | Asset tokenization, real-world assets | ⭐⭐⭐⭐⭐ |
| **@MakerDAO** | DAI protocol | Stablecoin, RWA integration | ⭐⭐⭐⭐⭐ |
| **@chainlink** | Oracle network | Data feeds, RWA infrastructure | ⭐⭐⭐⭐⭐ |

### **Research/Educational (Crypto-focused)**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@MessariCrypto** | Research firm | Institutional analysis, market data | ⭐⭐⭐⭐ |
| **@DeFiPulse** | Analytics platform | DeFi metrics, protocol tracking | ⭐⭐⭐⭐ |
| **@defiprime** | Education platform | DeFi tutorials, protocol guides | ⭐⭐⭐⭐ |

### **Technical Educators (No price noise)**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@evan_van_ness** | Newsletter author | Ethereum weekly updates | ⭐⭐⭐⭐ |
| **@sassal0x** | Technical analyst | Educational content, analysis | ⭐⭐⭐⭐ |
| **@tokenbrice** | DeFi educator | French DeFi education, protocols | ⭐⭐⭐⭐ |

**Total: 15 high-quality focused accounts**

---

## 📊 **SECONDARY KOL LIST: General Crypto Ecosystem**
*Broader crypto coverage including institutional and mainstream figures*

**Source:** `src/config.py` - **BROADER ECOSYSTEM COVERAGE**

### **Tech Leaders & CEOs**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@elonmusk** | Tesla/SpaceX CEO | Tech innovation, crypto commentary | ⭐⭐⭐ |
| **@michael_saylor** | MicroStrategy CEO | Bitcoin advocacy, corporate adoption | ⭐⭐⭐⭐ |
| **@cz_binance** | Binance CEO | Exchange operations, market trends | ⭐⭐⭐ |

### **Crypto Native Builders**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@VitalikButerin** | Ethereum founder | Technical insights (overlap with primary) | ⭐⭐⭐⭐⭐ |

### **Traders & Analysts**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@novogratz** | Galaxy Digital CEO | Institutional crypto, markets | ⭐⭐⭐⭐ |
| **@CryptoCobain** | Crypto trader | Market analysis, trading insights | ⭐⭐⭐ |
| **@CryptoBullish** | Content creator | General crypto content | ⭐⭐ |
| **@TheCryptoDog** | Crypto analyst | Market commentary, analysis | ⭐⭐⭐ |
| **@CryptoKaleo** | Technical analyst | TA, market predictions | ⭐⭐⭐ |
| **@KoroushAK** | Crypto trader | Trading strategies, market calls | ⭐⭐ |

### **Controversial/Critical Voices**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@SBF_FTX** | Former FTX CEO | ⚠️ **INACTIVE/PROBLEMATIC** | ❌ |
| **@peter_schiff** | Gold advocate | Bitcoin criticism, traditional finance | ⭐⭐ |

### **Political/Regulatory**
| Username | Role | Focus | Signal Quality |
|----------|------|--------|----------------|
| **@realDonaldTrump** | Former US President | Political crypto stance | ⭐⭐ |
| **@JoeBiden** | US President | Current political stance | ⭐⭐ |
| **@SECGov** | SEC official | Regulatory announcements | ⭐⭐⭐⭐ |
| **@CFTCgov** | CFTC official | Regulatory guidance | ⭐⭐⭐⭐ |

**Total: 16 accounts (1 problematic)**

---

## 🛡️ **BULLETPROOF COLLECTION RECOMMENDATIONS**

### **✅ RECOMMENDED FOR AUTOMATED COLLECTION:**
**Use the PRIMARY LIST (Focused Crypto/RWA)** because:
- ✅ **Higher signal-to-noise ratio**
- ✅ **Technical focus over speculation**
- ✅ **Educational content priority**
- ✅ **RWA expertise included**
- ✅ **Lower controversy/drama**

### **🔧 COLLECTION STRATEGY:**
```python
# Primary focused collection
PRIORITY_KOLS = [
    "VitalikButerin", "haydenzadams", "stani_kulechov",
    "centrifuge", "MakerDAO", "chainlink",
    "MessariCrypto", "DeFiPulse", "evan_van_ness"
]

# Secondary broader ecosystem (manual review)
BROADER_KOLS = [
    "michael_saylor", "novogratz", "SECGov", "CFTCgov"
]
```

### **⚠️ EXCLUDE FROM AUTOMATED COLLECTION:**
- **@SBF_FTX** - Account inactive/problematic
- **@realDonaldTrump** / **@JoeBiden** - High political noise
- **@elonmusk** - High off-topic noise
- **@peter_schiff** - Anti-Bitcoin bias

---

## 📊 **KOL PERFORMANCE METRICS**

### **Quality Scoring Criteria:**
- **⭐⭐⭐⭐⭐** - Essential (Protocol founders, core infrastructure)
- **⭐⭐⭐⭐** - High value (Educators, institutional research)
- **⭐⭐⭐** - Moderate value (Traders, analysts)
- **⭐⭐** - Low signal (High noise, controversial)
- **❌** - Exclude (Inactive, problematic)

### **Evaluation Metrics:**
1. **Crypto relevance ratio** - % of tweets about crypto/DeFi/RWA
2. **Technical depth** - Educational vs. speculative content
3. **Engagement quality** - Meaningful discussion vs. hype
4. **Consistency** - Regular posting, reliable insights

---

## 🎯 **USAGE IN XINFLUENCER SYSTEM**

### **For Bulletproof Collection (`safe_collection_script.py`):**
- **Use search queries** rather than user timelines
- **Focus on PRIMARY LIST** for highest quality
- **Manual review** for SECONDARY LIST additions

### **For Pipeline Collection (`data_collection_pipeline.py`):**
- **PRIMARY LIST** for automated daily collection
- **SECONDARY LIST** for weekly broader analysis
- **Exclude problematic accounts**

### **For H200 Deployment:**
- **Start with 5-10 top accounts** from PRIMARY LIST
- **Scale up based on API performance**
- **Monitor for signal quality degradation**

---

## 🔄 **MAINTENANCE SCHEDULE**

### **Monthly Review:**
- [ ] Check account activity status
- [ ] Evaluate content quality changes  
- [ ] Update signal quality ratings
- [ ] Add new high-quality accounts

### **Quarterly Analysis:**
- [ ] Performance analysis of each KOL
- [ ] Engagement metrics review
- [ ] Crypto relevance ratio assessment
- [ ] List optimization based on AI learning

---

## 📝 **CHANGE LOG**

**2025-07-21:** Initial comprehensive KOL list compilation
- Added focused crypto/RWA list (15 accounts)
- Added general crypto ecosystem list (16 accounts)
- Implemented quality scoring system
- Provided bulletproof collection recommendations
