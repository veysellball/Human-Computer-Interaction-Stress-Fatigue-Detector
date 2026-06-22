import pandas as pd
import numpy as np

class RealTimeFeatureExtractor:
    def __init__(self, top_n_apps=None):
        # We use the previous state to calculate rolling and delta features
        self.past_3_blocks = []
        # Pre-configured Top 7 apps from training process
        self.top_n_apps = top_n_apps or [
            'chrome', 'explorer', 'code', 'devenv', 'msedge', 'teams', 'opera'
        ]

    def process_payload(self, payload: dict) -> dict:
        """
        Process exactly 3-minute worth of raw telemetry events into the 47 ML features.
        
        payload schema:
        {
           "mouse_events": [...],      # x, y, type('move', 'click'), time(ms)
           "keyboard_events": [...],   # key, type('press', 'release'), time(ms)
           "window_events": [...]      # app_name, time(ms)
        }
        """
        mouse_events = payload.get("mouse_events", [])
        key_events = payload.get("keyboard_events", [])
        window_events = payload.get("window_events", [])
        
        # 1. Base aggregations
        block_features = self._extract_base_features(mouse_events, key_events, window_events)
        
        # 2. Append to history for rolling features
        self.past_3_blocks.append(block_features)
        if len(self.past_3_blocks) > 3:
            self.past_3_blocks.pop(0)
            
        # 3. Calculate deltas and rollings using self.past_3_blocks
        final_features = self._calculate_rolling_features(block_features)
        
        return final_features

    def _extract_base_features(self, m_events, k_events, w_events):
        feats = {}
        
        # MOUSE
        m_df = pd.DataFrame(m_events)
        if not m_df.empty:
            m_df['dt'] = m_df['time'].diff() / 1000.0  # seconds
            m_df['dx'] = m_df.get('x', pd.Series(dtype=float)).diff()
            m_df['dy'] = m_df.get('y', pd.Series(dtype=float)).diff()
            m_df['dist'] = np.sqrt(m_df['dx']**2 + m_df['dy']**2).fillna(0)
            move_mask = m_df['type'] == 'move'
            m_df.loc[~move_mask, 'dist'] = 0.0
            m_df['speed'] = np.where(m_df['dt'] > 0, m_df['dist'] / m_df['dt'], 0.0)
            
            feats['mouse_move_count'] = move_mask.sum()
            feats['mouse_click_count'] = (m_df['type'] == 'click').sum()
            feats['mouse_path_length_px'] = m_df['dist'].sum()
            feats['mouse_avg_speed_px_s'] = m_df['speed'].mean() if move_mask.sum() > 0 else 0
            feats['mouse_std_speed_px_s'] = m_df['speed'].std() if move_mask.sum() > 1 else 0
            feats['mouse_idle_time_ms'] = m_df.loc[m_df['dt'] > 1.0, 'dt'].sum() * 1000
            
            direct_dist = np.sqrt((m_df['x'].iloc[-1] - m_df['x'].iloc[0])**2 + (m_df['y'].iloc[-1] - m_df['y'].iloc[0])**2) if len(m_df) > 1 else 0
            feats['mouse_path_straightness'] = min(direct_dist / feats['mouse_path_length_px'], 1.0) if feats['mouse_path_length_px'] > 0 else 0
        else:
            for c in ['mouse_move_count', 'mouse_click_count', 'mouse_path_length_px', 'mouse_avg_speed_px_s', 'mouse_std_speed_px_s', 'mouse_idle_time_ms', 'mouse_path_straightness']:
                feats[c] = 0.0

        # KEYBOARD
        k_df = pd.DataFrame(k_events)
        if not k_df.empty:
            presses = k_df[k_df['type'] == 'press']
            feats['keydown_count'] = len(presses)
            feats['typing_rate_kps'] = len(presses) / 180.0 # 3 min = 180s
            
            special_keys = ['shift', 'ctrl', 'alt', 'enter', 'space', 'backspace']
            feats['special_key_count'] = presses['key'].isin(special_keys).sum()
            feats['backspace_count'] = presses['key'].isin(['backspace', 'delete']).sum()
            feats['backspace_error_ratio'] = feats['backspace_count'] / len(presses) if len(presses) > 0 else 0
            
            feats['inter_key_latency_mean_ms'] = presses['time'].diff().mean() if len(presses) > 1 else 0
            feats['key_dwell_time_mean_ms'] = 100 # Mock calculation if release events aren't perfectly paired
            feats['key_dwell_time_std_ms'] = 0
            feats['long_pause_count'] = (presses['time'].diff() > 2000).sum() if len(presses) > 1 else 0
        else:
            for c in ['keydown_count', 'typing_rate_kps', 'special_key_count', 'backspace_count', 'backspace_error_ratio', 'inter_key_latency_mean_ms', 'key_dwell_time_mean_ms', 'key_dwell_time_std_ms', 'long_pause_count']:
                feats[c] = 0.0
                
        # CONTEXT
        w_df = pd.DataFrame(w_events)
        if not w_df.empty:
            feats['window_switch_count'] = (w_df['app_name'] != w_df['app_name'].shift()).sum()
            feats['unique_app_count'] = w_df['app_name'].nunique()
            dominant_app = w_df['app_name'].mode()[0] if not w_df.empty else 'none'
        else:
            feats['window_switch_count'] = 0
            feats['unique_app_count'] = 0
            dominant_app = 'none'
            
        for app in self.top_n_apps:
            feats[f'app_{app}'] = 1 if dominant_app.lower() == app else 0
        feats['app_other'] = 1 if dominant_app.lower() not in self.top_n_apps and dominant_app != 'none' else 0
        
        # Time of day mockup - assume backend sets this natively based on request time
        feats['daylight_morning'] = 1
        feats['daylight_afternoon'] = 0
        feats['daylight_evening'] = 0
        
        # RATIOS
        feats['activity_ratio'] = 1 if feats['mouse_move_count'] > 0 or feats['keydown_count'] > 0 else 0
        feats['mouse_speed_cv'] = feats['mouse_std_speed_px_s'] / feats['mouse_avg_speed_px_s'] if feats['mouse_avg_speed_px_s'] > 0 else 0
        feats['typing_burst_count'] = 1 if feats['typing_rate_kps'] > 0 else 0
        feats['idle_to_active_ratio'] = feats['mouse_idle_time_ms'] / (feats['mouse_move_count'] + 1)
        feats['click_per_move_ratio'] = feats['mouse_click_count'] / feats['mouse_move_count'] if feats['mouse_move_count'] > 0 else 0
        
        return feats

    def _calculate_rolling_features(self, current_feats):
        final = current_feats.copy()
        rolling_targets = [
            'typing_rate_kps', 'inter_key_latency_mean_ms', 'key_dwell_time_mean_ms',
            'mouse_avg_speed_px_s', 'mouse_idle_time_ms', 'activity_ratio'
        ]
        
        if len(self.past_3_blocks) >= 2:
            prev = self.past_3_blocks[-2]
            for t in rolling_targets:
                final[f'{t}_delta'] = final[t] - prev[t]
        else:
            for t in rolling_targets:
                final[f'{t}_delta'] = 0.0
                
        if len(self.past_3_blocks) > 0:
            for t in rolling_targets:
                vals = [b[t] for b in self.past_3_blocks]
                final[f'{t}_rmean3'] = np.mean(vals)
                final[f'{t}_rstd3'] = np.std(vals) if len(vals) > 1 else 0.0
        else:
            for t in rolling_targets:
                final[f'{t}_rmean3'] = final[t]
                final[f'{t}_rstd3'] = 0.0
                
        return final
