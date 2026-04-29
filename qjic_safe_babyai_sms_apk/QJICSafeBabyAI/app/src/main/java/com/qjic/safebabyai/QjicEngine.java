package com.qjic.safebabyai;

import android.content.SharedPreferences;

import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

public class QjicEngine {
    public static final String[] LABELS = {"정상", "의심", "위험"};
    public static final int SAFE = 0;
    public static final int WARN = 1;
    public static final int DANGER = 2;
    public static final int DIM = 384;
    public static final double LR = 0.26;
    public static final double DECAY = 0.0005;

    public double[][] W_base = new double[3][DIM];
    public double[][] W_adapter = new double[3][DIM];
    public double[][] W_personal = new double[3][DIM];
    public double[] b_personal = new double[3];
    public int predictCount;
    public int learnCount;
    public double lastL;

    private static final String[][] PHASE_A = {
            {"엄마 오늘 저녁 먹자", "병원 예약 접수가 완료되었습니다", "배송 상품 도착 예정입니다"},
            {"고객님 본인 인증 확인 부탁드립니다", "해외 접속 지연으로 추가 인증 필요", "결제 내역 확인해 주세요"},
            {"검찰청 사건 연루 계좌 보호 위해 송금", "안전계좌로 즉시 이체하세요", "링크 클릭 후 개인정보 입력"}
    };
    private static final String[][] PHASE_B = {
            {"아버지 약 드실 시간입니다", "복지관 차량 방문 일정 변경되었습니다", "좋은 하루 보내세요"},
            {"보안 강화를 위한 명의 확인", "금융 보안 긴급 안내", "계정 보호를 위해 입력해 주세요"},
            {"정부지원금 환급금 수령 링크 접속", "저금리 대출 승인 수수료 입금하세요", "가족 사고가 나서 병원비 급히 보내"}
    };

    public static final String[][] TEST_SET = {
            {"정상", "엄마 오늘 회의 끝나고 전화할게"},
            {"정상", "예약 진료 일정이 변경되었습니다"},
            {"의심", "고객님 계정 보안 확인을 위해 인증 필요"},
            {"의심", "해외 접속 감지로 추가 인증 요청"},
            {"위험", "안전계좌로 지금 바로 송금하세요"},
            {"위험", "링크 접속 후 카드정보 입력 바랍니다"}
    };

    private static final String[] DANGER_WORDS = {"송금", "이체", "안전계좌", "계좌 보호", "돈을 보내", "입금하세요", "수수료", "보증금", "환급금", "병원비", "링크 클릭", "링크 접속", "앱 설치", "개인정보 입력", "카드정보 입력", "검찰청", "수사기관", "국세청", "사건에 연루", "정지되었습니다", "차단 해제", "급히", "즉시", "대출 승인", "저금리 대출", "정부지원금", "소액결제", "가족 사고", "사고가 나서", "다쳐서", "주소 오류", "청첩장", "미납 요금"};
    private static final String[] WARN_WORDS = {"확인", "본인", "인증", "보안", "계정", "결제", "내역", "긴급 안내", "명의 확인", "지연", "해외 접속", "추가 인증", "금융 보안", "고객님", "입력해 주세요"};
    private static final String[] SAFE_WORDS = {"엄마", "아빠", "아버지", "어머니", "저녁", "회의", "약속", "예약", "진료", "병원", "복지관", "차량", "방문", "변경되었습니다", "배송", "도착 예정", "상품", "생일 축하", "좋은 하루", "날씨", "정상 입금", "약 드실", "접수가 완료"};

    private final Random random = new Random(7);

    public QjicEngine() { buildBase(); buildAdapter(); }
    public static int idx(String label) { for (int i=0;i<LABELS.length;i++) if (LABELS[i].equals(label)) return i; return SAFE; }

    public Result predict(String text) {
        double[] x = vectorize(text);
        double[] raw = new double[3];
        for (int c = 0; c < 3; c++) raw[c] = dot(W_base[c], x) + dot(W_adapter[c], x) + dot(W_personal[c], x) + b_personal[c];
        double[] s = new double[3]; for (int c=0;c<3;c++) s[c]=Math.tanh(raw[c]);
        int p=0; if(s[1]>s[p])p=1; if(s[2]>s[p])p=2;
        double t=-9,sec=-9; for(double v:s){ if(v>t){sec=t;t=v;} else if(v>sec) sec=v; }
        Result r = new Result(); r.pred=p; r.S=s; r.confidence=clamp((t-sec+1.0)/2.0,0,1); predictCount++; return r;
    }

    public double train(String text, int label) {
        double[] x = vectorize(text); Result pr = predict(text);
        double[] it = target(label); double[] l = new double[3]; for(int c=0;c<3;c++) l[c]=it[c]-pr.S[c];
        for (int c=0;c<3;c++) { for (int i=0;i<DIM;i++) W_personal[c][i]=(1-DECAY)*W_personal[c][i]+LR*l[c]*x[i]; b_personal[c]=(1-DECAY)*b_personal[c]+LR*0.22*l[c]; }
        double m = meanAbs(l); learnCount++; lastL=m; if (measureLOld()>0.34) weakReplay(); return m;
    }

    public double measureLOld() {
        double sum=0; int n=0;
        for(int c=0;c<3;c++) for(String t:PHASE_A[c]) {sum+=residual(t,c); n++;}
        for(int c=0;c<3;c++) for(String t:PHASE_B[c]) {sum+=residual(t,c); n++;}
        return n==0?0:sum/n;
    }
    private double residual(String text,int label){ Result p=predict(text); double[] l=target(label); for(int i=0;i<3;i++) l[i]-=p.S[i]; return meanAbs(l); }
    public void weakReplay() {
        for (int k=0;k<3;k++) {
            int c = random.nextInt(3); String[] arr = random.nextBoolean()?PHASE_A[c]:PHASE_B[c]; String t = arr[random.nextInt(arr.length)];
            double[] x = vectorize(t); Result p = predict(t); double[] l = target(c); for(int i=0;i<3;i++) l[i]-=p.S[i];
            for(int cc=0;cc<3;cc++){ for(int i=0;i<DIM;i++) W_personal[cc][i]+=0.035*l[cc]*x[i]; b_personal[cc]+=0.035*0.1*l[cc]; }
        }
    }

    public void resetPersonal() { W_personal = new double[3][DIM]; b_personal = new double[3]; learnCount = 0; lastL = 0; }

    public void save(SharedPreferences prefs) {
        SharedPreferences.Editor e=prefs.edit(); e.putInt("predictCount",predictCount); e.putInt("learnCount",learnCount); e.putFloat("lastL",(float)lastL);
        for(int c=0;c<3;c++){ StringBuilder sb=new StringBuilder(); for(int i=0;i<DIM;i++){ if(i>0)sb.append(','); sb.append(W_personal[c][i]); } e.putString("wp_"+c,sb.toString()); e.putFloat("bp_"+c,(float)b_personal[c]); }
        e.apply();
    }
    public void load(SharedPreferences prefs) {
        predictCount=prefs.getInt("predictCount",0); learnCount=prefs.getInt("learnCount",0); lastL=prefs.getFloat("lastL",0);
        for(int c=0;c<3;c++){ String s=prefs.getString("wp_"+c,""); if(s!=null&&!s.isEmpty()){ String[] sp=s.split(","); for(int i=0;i<Math.min(sp.length,DIM);i++) try{W_personal[c][i]=Double.parseDouble(sp[i]);}catch(Exception ignored){} } b_personal[c]=prefs.getFloat("bp_"+c,0); }
    }

    private void buildBase(){ ingest(SAFE_WORDS,SAFE,0.95); ingest(WARN_WORDS,WARN,0.95); ingest(DANGER_WORDS,DANGER,1.1); }
    private void buildAdapter(){ for(int c=0;c<3;c++){ for(String t:PHASE_A[c]) addSample(t,c,0.6); for(String t:PHASE_B[c]) addSample(t,c,0.5);} }
    private void ingest(String[] ws,int label,double w){ for(String s:ws){ double[] x=vectorize(s); for(int i=0;i<DIM;i++) W_base[label][i]+=w*x[i]; } }
    private void addSample(String t,int label,double w){ double[] x=vectorize(t); for(int i=0;i<DIM;i++) W_adapter[label][i]+=w*x[i]; }

    private double[] vectorize(String text) {
        String n = normalize(text); ArrayList<String> tk = tokenize(n); double[] v = new double[DIM];
        for(String t:tk){ int h=positiveHash("w:"+t)%DIM; v[h]+=1.0; for(int g=2;g<=4;g++) if(t.length()>=g) for(int i=0;i<=t.length()-g;i++) v[positiveHash("n:"+t.substring(i,i+g))%DIM]+=0.42; }
        double s=0; for(double d:v) s+=d*d; s=Math.sqrt(Math.max(s,1e-12)); for(int i=0;i<DIM;i++) v[i]/=s; return v;
    }
    private ArrayList<String> tokenize(String text){ ArrayList<String> out=new ArrayList<>(); for(String p:text.split("\\s+")) if(!p.isEmpty()) out.add(p); return out; }
    private String normalize(String text){ if(text==null) return ""; return text.toLowerCase(Locale.KOREA).replaceAll("[^0-9a-z가-힣\\s]"," ").replaceAll("\\s+"," ").trim(); }
    private int positiveHash(String s){ long h=0x811c9dc5L; for(int i=0;i<s.length();i++){ h^=s.charAt(i); h*=0x01000193L; h&=0xffffffffL;} return (int)(h & 0x7fffffff); }
    private double[] target(int label){ double[] t={-0.85,-0.85,-0.85}; t[label]=0.95; return t; }
    private double dot(double[] a,double[] b){ double s=0; for(int i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
    private double meanAbs(double[] a){ double s=0; for(double v:a) s+=Math.abs(v); return s/a.length; }
    private double clamp(double v,double lo,double hi){ return Math.max(lo,Math.min(hi,v)); }

    public static class Result { public int pred; public double confidence; public double[] S; }
}
