package applicationpermissions;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;

public class AndroidPermissions {
    private Activity mUnityActivity;
    protected static final String TAG = "mUnityActivity";

    // Must call in unity to initialize Activity
    public void setUnityActivity(Activity unityActivity) {
        this.mUnityActivity = unityActivity;
    }

    public void requestExternalStorage() {
        // Request permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                Uri uri = Uri.fromParts("package", mUnityActivity.getPackageName(), null);
                intent.setData(uri);
                mUnityActivity.startActivity(intent);
            } else {
                // The user has granted full access permissions, and you can proceed with the relevant operation
            }
        } else {
            // For Android 10 and lower versions, there is no need to request the MANAGE_EXTERNAL_STORAGE permission separately
        }
    }
}
