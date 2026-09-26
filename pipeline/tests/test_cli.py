from horizon.cli import build_parser, main


def test_parser_exposes_doctor_command():
    args = build_parser().parse_args(["doctor"])
    assert args.command == "doctor"
    assert callable(args.handler)


def test_ball_command_is_discovered():
    args = build_parser().parse_args(["ball", "--match-id", "demo"])
    assert args.match_id == "demo"
    assert callable(args.handler)


def test_doctor_reports_tools_without_leaking_secrets(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-value")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ffmpeg:" in out
    assert "OPENROUTER_API_KEY: set" in out
    assert "super-secret-value" not in out
