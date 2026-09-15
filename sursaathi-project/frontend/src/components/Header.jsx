import { useNavigate } from "react-router-dom";
import Logo from "./Logo";

export default function Header() {
  const navigate = useNavigate();
  return (
    <header className="site-header">
      <div className="brand" onClick={() => navigate("/")}>
        <Logo size={32} />
        <span className="wordmark">SurSaathi</span>
      </div>
    </header>
  );
}
