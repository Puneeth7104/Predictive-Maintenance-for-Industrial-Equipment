"""Run the whole pipeline: data -> model -> risk scores -> static dashboard."""
from . import build_static_dashboard, generate_data, risk_scoring, train


def main():
    generate_data.main()
    train.train_and_save()
    risk_scoring.run()
    build_static_dashboard.build()
    print("[pipeline] finished")


if __name__ == "__main__":
    main()
