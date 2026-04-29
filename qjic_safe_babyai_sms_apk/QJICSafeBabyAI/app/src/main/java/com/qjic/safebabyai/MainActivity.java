package com.qjic.safebabyai;

import android.Manifest;
import android.app.Activity;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {
    private SharedPreferences prefs; private QjicEngine engine; private EditText input; private TextView result, detail, status, log; private String currentText="";
    @Override protected void onCreate(Bundle savedInstanceState) { super.onCreate(savedInstanceState); prefs=getSharedPreferences("qjic_prefs",MODE_PRIVATE); engine=new QjicEngine(); engine.load(prefs); setContentView(buildUi()); requestRuntimePermissions(); updateStatus(); }
    private ScrollView buildUi(){ ScrollView sv=new ScrollView(this); LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setPadding(24,24,24,24); TextView h=new TextView(this); h.setText("QJIC Safe BabyAI"); h.setTextSize(28); h.setBackgroundColor(0xFF111111); h.setTextColor(0xFFFFFFFF); h.setPadding(20,24,20,24); h.setGravity(Gravity.CENTER); root.addView(h); TextView d=new TextView(this); d.setText("온디바이스 보이스피싱/위험문자 자가학습 AI"); d.setTextSize(20); d.setPadding(0,20,0,20); root.addView(d); input=new EditText(this); input.setHint("문자 내용을 입력하세요"); input.setTextSize(20); root.addView(input); root.addView(btn("판정하기",v->doPredict(input.getText().toString()))); LinearLayout ex=row(); ex.addView(btn("정상 예시",v->input.setText("엄마 오늘 저녁 약속 있어"))); ex.addView(btn("의심 예시",v->input.setText("고객님 본인 인증을 위해 확인"))); ex.addView(btn("위험 예시",v->input.setText("안전계좌로 즉시 송금하세요"))); root.addView(ex); result=tv(22); detail=tv(18); root.addView(result); root.addView(detail); LinearLayout fb=row(); fb.addView(btn("정상입니다",v->learn(QjicEngine.SAFE))); fb.addView(btn("의심입니다",v->learn(QjicEngine.WARN))); fb.addView(btn("위험입니다",v->learn(QjicEngine.DANGER))); root.addView(fb); status=tv(18); root.addView(status); root.addView(btn("내장 테스트 실행",v->runSelfTest())); root.addView(btn("개인학습 초기화",v->{engine.resetPersonal(); engine.save(prefs); appendLog("개인학습 초기화 완료"); updateStatus();})); log=tv(16); root.addView(log); sv.addView(root); return sv; }
    private void doPredict(String t){ currentText=t; QjicEngine.Result r=engine.predict(t); result.setText("판정 결과: "+QjicEngine.LABELS[r.pred]+" (신뢰도 "+String.format("%.3f",r.confidence)+")"); detail.setText("S값: ["+f(r.S[0])+", "+f(r.S[1])+", "+f(r.S[2])+"] / 최근 |L|="+f(engine.lastL)+" / L_old="+f(engine.measureLOld())); engine.save(prefs); updateStatus(); }
    private void learn(int label){ String t=input.getText().toString(); if(t.trim().isEmpty()) t=currentText; double ml=engine.train(t,label); QjicEngine.Result r=engine.predict(t); engine.save(prefs); appendLog("학습 반영: "+QjicEngine.LABELS[label]+", mean|L|="+f(ml)+" -> 재판정="+QjicEngine.LABELS[r.pred]); doPredict(t); }
    private void runSelfTest(){ int hit=0; StringBuilder sb=new StringBuilder("[SelfTest]\n"); for(String[] s:QjicEngine.TEST_SET){ QjicEngine.Result r=engine.predict(s[1]); int gt=QjicEngine.idx(s[0]); if(r.pred==gt) hit++; sb.append(s[0]).append(" | ").append(s[1]).append(" => ").append(QjicEngine.LABELS[r.pred]).append("\n"); } sb.append("정확도: ").append(hit).append("/").append(QjicEngine.TEST_SET.length); appendLog(sb.toString()); engine.save(prefs); updateStatus(); }
    private void updateStatus(){ String sdr=prefs.getString("last_sms_sender","-"); String body=prefs.getString("last_sms_body","-"); String rs=prefs.getString("last_sms_result","-"); float cf=prefs.getFloat("last_sms_confidence",0); status.setText("QJIC 상태\n판정 횟수: "+engine.predictCount+"\n개인 학습 횟수: "+engine.learnCount+"\n평균 |L|: "+f(engine.lastL)+"\nL_old: "+f(engine.measureLOld())+"\n마지막 수신 발신자: "+sdr+"\n마지막 수신 내용: "+body+"\n마지막 수신 판정: "+rs+"\n마지막 수신 신뢰도: "+f(cf)+"\n구조: W_base + W_adapter + W_personal / I = S + L"); }
    private void requestRuntimePermissions(){ if(Build.VERSION.SDK_INT>=23&&checkSelfPermission(Manifest.permission.RECEIVE_SMS)!=PackageManager.PERMISSION_GRANTED) requestPermissions(new String[]{Manifest.permission.RECEIVE_SMS},11); if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED) requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},12); }
    private Button btn(String t, android.view.View.OnClickListener l){ Button b=new Button(this); b.setText(t); b.setTextSize(18); b.setOnClickListener(l); return b; } private LinearLayout row(){ LinearLayout l=new LinearLayout(this); l.setOrientation(LinearLayout.HORIZONTAL); return l; } private TextView tv(int sz){ TextView t=new TextView(this); t.setTextSize(sz); t.setPadding(0,18,0,8); return t; } private void appendLog(String s){ log.setText(s+"\n\n"+log.getText()); } private String f(double d){ return String.format("%.3f",d); }
}
