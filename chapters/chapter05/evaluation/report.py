"""Read-only legacy report helpers; current paper plots use each local figure/plot.py."""
from chapters.chapter05._runtime import expose
expose('chapter5_full_spend_20261104.report', globals())
if __name__ == '__main__':
    raise SystemExit('Use the chapter figure plot.py entry for current paper graphics; this module exposes source helpers only.')
