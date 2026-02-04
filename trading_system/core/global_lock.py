"""
Global Account Lock
防止多個策略同時下單造成race condition
"""
import fcntl
import time
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/futures_account_lock"
DEFAULT_TIMEOUT = 10  # 秒

class LockTimeoutError(Exception):
    """鎖獲取超時"""
    pass

@contextmanager
def global_account_lock(lock_path=LOCK_FILE, timeout_sec=DEFAULT_TIMEOUT):
    """
    全局帳戶鎖
    
    用法:
        with global_account_lock():
            # 獲取帳戶快照
            # 評估Gate
            # 下單
            pass
    """
    lock_file = None
    acquired = False
    
    try:
        # 確保鎖文件存在
        Path(lock_path).touch()
        
        # 打開鎖文件
        lock_file = open(lock_path, 'r')
        
        # 嘗試獲取鎖
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                logger.debug("✅ 獲取全局鎖")
                break
            except IOError:
                time.sleep(0.1)
        
        if not acquired:
            raise LockTimeoutError(f"無法在{timeout_sec}秒內獲取全局鎖")
        
        # 執行被保護的代碼
        yield
        
    finally:
        # 釋放鎖
        if acquired and lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            logger.debug("🔓 釋放全局鎖")
        
        if lock_file:
            lock_file.close()

# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("測試全局鎖...")
    
    with global_account_lock():
        print("  ✅ 獲取鎖成功")
        time.sleep(1)
        print("  ✅ 執行臨界區代碼")
    
    print("  ✅ 釋放鎖成功")
    print("\n測試完成!")
