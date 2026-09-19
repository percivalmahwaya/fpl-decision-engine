"""Repair the markdown block the shell mangled into real newlines."""
import io

p = 'app.py'
s = io.open(p, encoding='utf-8').read()

broken = '''                st.markdown(
                    f"**Out** {m['out']} &nbsp; {m['out_price']:.1f}m &nbsp; "
                    f"{m['out_ep']:.2f} xP
"
                    f"**In** &nbsp;&nbsp;{m['in']} ({m['in_club']}) &nbsp; "
                    f"{m['in_price']:.1f}m &nbsp; {m['in_ep']:.2f} xP{pen}
"
                    f"**Money** {m['cost']:+.1f}m, leaving {m['bank_after']:.1f}m "
                    "in the bank", unsafe_allow_html=True)'''

fixed = '''                st.markdown(
                    f"**Out** {m['out']} &nbsp; {m['out_price']:.1f}m &nbsp; "
                    f"{m['out_ep']:.2f} xP<br>"
                    f"**In** &nbsp;&nbsp;{m['in']} ({m['in_club']}) &nbsp; "
                    f"{m['in_price']:.1f}m &nbsp; {m['in_ep']:.2f} xP{pen}<br>"
                    f"**Money** {m['cost']:+.1f}m, leaving "
                    f"{m['bank_after']:.1f}m in the bank",
                    unsafe_allow_html=True)'''

assert broken in s, 'broken block not found'
io.open(p, 'w', encoding='utf-8').write(s.replace(broken, fixed, 1))
print('markdown block repaired')
