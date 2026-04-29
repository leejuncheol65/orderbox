package com.qjic.safebabyai;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.telephony.SmsMessage;

public class SmsReceiver extends BroadcastReceiver {
    private static final String CH_ID = "qjic_sms_alert";
    @Override public void onReceive(Context context, Intent intent) {
        if (intent == null || !"android.provider.Telephony.SMS_RECEIVED".equals(intent.getAction())) return;
        Bundle bundle = intent.getExtras(); if (bundle == null) return;
        Object[] pdus = (Object[]) bundle.get("pdus"); String format = bundle.getString("format"); if (pdus == null || pdus.length == 0) return;
        StringBuilder body = new StringBuilder(); String sender = "unknown";
        for (Object pdu : pdus) { SmsMessage msg = Build.VERSION.SDK_INT >= 23 ? SmsMessage.createFromPdu((byte[]) pdu, format) : SmsMessage.createFromPdu((byte[]) pdu); if (msg != null) { if (msg.getDisplayOriginatingAddress() != null) sender = msg.getDisplayOriginatingAddress(); body.append(msg.getMessageBody()); } }
        SharedPreferences prefs = context.getSharedPreferences("qjic_prefs", Context.MODE_PRIVATE);
        QjicEngine engine = new QjicEngine(); engine.load(prefs); QjicEngine.Result r = engine.predict(body.toString()); engine.save(prefs);
        prefs.edit().putString("last_sms_sender", sender).putString("last_sms_body", body.toString()).putString("last_sms_result", QjicEngine.LABELS[r.pred]).putFloat("last_sms_confidence", (float) r.confidence).apply();
        if (r.pred == QjicEngine.DANGER || r.pred == QjicEngine.WARN) showNotification(context, sender, body.toString(), r);
    }
    private void showNotification(Context c, String sender, String body, QjicEngine.Result r) {
        NotificationManager nm = (NotificationManager) c.getSystemService(Context.NOTIFICATION_SERVICE); if (nm == null) return;
        if (Build.VERSION.SDK_INT >= 26) { NotificationChannel ch = new NotificationChannel(CH_ID, "QJIC SMS Alert", NotificationManager.IMPORTANCE_HIGH); nm.createNotificationChannel(ch); }
        Intent i = new Intent(c, MainActivity.class); i.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pi = PendingIntent.getActivity(c, 101, i, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        String title = (r.pred == QjicEngine.DANGER) ? "QJIC 위험 문자 감지" : "QJIC 의심 문자 감지";
        String txt = "발신자:" + sender + " / 판정:" + QjicEngine.LABELS[r.pred] + " / 신뢰도:" + String.format("%.2f", r.confidence) + " / " + body.substring(0, Math.min(24, body.length()));
        Notification.Builder b = Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(c, CH_ID) : new Notification.Builder(c);
        b.setContentTitle(title).setContentText(txt).setSmallIcon(android.R.drawable.stat_notify_error).setContentIntent(pi).setAutoCancel(true);
        nm.notify(2001, b.build());
    }
}
